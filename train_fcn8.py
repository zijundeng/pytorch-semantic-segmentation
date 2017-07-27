import os
import random

import torch
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from tensorboard import SummaryWriter
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader

import utils.simul_transforms as simul_transforms
import utils.transforms as expanded_transforms
from config import ckpt_path
from datasets.cityscapes import CityScapes
from datasets.cityscapes.config import num_classes, ignored_label
from datasets.cityscapes.utils import colorize_mask
from models import FCN8ResNet
from utils.io import rmrf_mkdir
from utils.loss import CrossEntropyLoss2d
from utils.training import calculate_mean_iu

cudnn.benchmark = True
exp_name = 'fcn8resnet_cityscapes224*448'
writer = SummaryWriter('exp/' + exp_name)
pil_to_tensor = standard_transforms.ToTensor()
train_record = {'best_val_loss': 1e20, 'corr_mean_iu': 0, 'corr_epoch': -1}

train_args = {
    'batch_size': 24,
    'epoch_num': 800,  # I stop training only when val loss doesn't seem to decrease anymore, so just set a large value.
    'pretrained_lr': 1e-3,  # used for the pretrained layers of model
    'new_lr': 1e-2,  # used for the newly added layers of model
    'weight_decay': 5e-4,
    'snapshot': '',  # empty string denotes initial training, otherwise it should be a string of snapshot name
    'print_freq': 50,
    'input_size': (224, 448),  # (height, width)
}

val_args = {
    'batch_size': 8,
    'tensorboard_img_sample_rate': 0.15
}


def main():
    net = FCN8ResNet(num_classes=num_classes).cuda()
    if len(train_args['snapshot']) == 0:
        curr_epoch = 0
    else:
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, train_args['snapshot'])))
        split_snapshot = train_args['snapshot'].split('_')
        curr_epoch = int(split_snapshot[1])
        train_record['best_val_loss'] = float(split_snapshot[3])
        train_record['corr_mean_iu'] = float(split_snapshot[6])
        train_record['corr_epoch'] = curr_epoch

    net.train()

    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_simul_transform = simul_transforms.Compose([
        simul_transforms.Scale(int(train_args['input_size'][0] / 0.875)),
        simul_transforms.RandomCrop(train_args['input_size']),
        simul_transforms.RandomHorizontallyFlip()
    ])
    val_simul_transform = simul_transforms.Compose([
        simul_transforms.Scale(int(train_args['input_size'][0] / 0.875)),
        simul_transforms.CenterCrop(train_args['input_size'])
    ])
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    target_transform = standard_transforms.Compose([
        expanded_transforms.MaskToTensor(),
        expanded_transforms.ChangeLabel(ignored_label, num_classes - 1)
    ])
    restore_transform = standard_transforms.Compose([
        expanded_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])

    train_set = CityScapes('train', simul_transform=train_simul_transform, transform=img_transform,
                           target_transform=target_transform)
    train_loader = DataLoader(train_set, batch_size=train_args['batch_size'], num_workers=16, shuffle=True)
    val_set = CityScapes('val', simul_transform=val_simul_transform, transform=img_transform,
                         target_transform=target_transform)
    val_loader = DataLoader(val_set, batch_size=val_args['batch_size'], num_workers=16, shuffle=False)

    weight = torch.ones(num_classes)
    weight[num_classes - 1] = 0
    criterion = CrossEntropyLoss2d(weight).cuda()

    # don't use weight_decay for bias
    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if
                    name[-4:] == 'bias' and 'fconv' in name],
         'lr': train_args['new_lr']},
        {'params': [param for name, param in net.named_parameters() if
                    name[-4:] != 'bias' and 'fconv' in name],
         'lr': train_args['new_lr'], 'weight_decay': train_args['weight_decay']},
        {'params': [param for name, param in net.named_parameters() if
                    name[-4:] == 'bias' and 'fconv' not in name],
         'lr': train_args['pretrained_lr']},
        {'params': [param for name, param in net.named_parameters() if
                    name[-4:] != 'bias' and 'fconv' not in name],
         'lr': train_args['pretrained_lr'], 'weight_decay': train_args['weight_decay']}
    ], momentum=0.9, nesterov=True)

    if len(train_args['snapshot']) > 0:
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, 'opt_' + train_args['snapshot'])))
        optimizer.param_groups[0]['lr'] = train_args['new_lr']
        optimizer.param_groups[1]['lr'] = train_args['new_lr']
        optimizer.param_groups[2]['lr'] = train_args['pretrained_lr']
        optimizer.param_groups[3]['lr'] = train_args['pretrained_lr']

    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)
    if not os.path.exists(os.path.join(ckpt_path, exp_name)):
        os.mkdir(os.path.join(ckpt_path, exp_name))

    for epoch in range(curr_epoch, train_args['epoch_num']):
        train(train_loader, net, criterion, optimizer, epoch)
        validate(val_loader, net, criterion, optimizer, epoch, restore_transform)


def train(train_loader, net, criterion, optimizer, epoch):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % train_args['print_freq'] == 0:
            outputs = outputs[:, :num_classes - 1, :, :]
            prediction = outputs.data.max(1)[1].squeeze_(1).cpu().numpy()
            mean_iu = calculate_mean_iu(prediction, labels.data.cpu().numpy(), num_classes)

            print '[epoch %d], [iter %d], [training loss %.4f], [mean_iu %.4f]' % (epoch + 1, i + 1, loss.data[0], mean_iu)


def validate(val_loader, net, criterion, optimizer, epoch, restore):
    net.eval()
    criterion.cpu()
    input_batches = []
    output_batches = []
    label_batches = []

    for vi, data in enumerate(val_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs, volatile=True).cuda()
        labels = Variable(labels, volatile=True).cuda()

        outputs = net(inputs)

        input_batches.append(inputs.cpu().data)
        output_batches.append(outputs.cpu())
        label_batches.append(labels.cpu())

    input_batches = torch.cat(input_batches)
    output_batches = torch.cat(output_batches)
    label_batches = torch.cat(label_batches)
    val_loss = criterion(output_batches, label_batches)
    val_loss = val_loss.data[0]

    output_batches = output_batches.cpu().data[:, :num_classes - 1, :, :]
    label_batches = label_batches.cpu().data.numpy()
    prediction_batches = output_batches.max(1)[1].squeeze_(1).numpy()

    mean_iu = calculate_mean_iu(prediction_batches, label_batches, num_classes)

    writer.add_scalar('loss', val_loss, epoch + 1)
    writer.add_scalar('mean_iu', mean_iu, epoch + 1)

    if val_loss < train_record['best_val_loss']:
        train_record['best_val_loss'] = val_loss
        train_record['corr_epoch'] = epoch + 1
        train_record['corr_mean_iu'] = mean_iu
        snapshot_name = 'epoch_%d_loss_%.4f_mean_iu_%.4f_lr_%.8f' % (
            epoch + 1, val_loss, mean_iu, train_args['new_lr'])
        torch.save(net.state_dict(), os.path.join(
            ckpt_path, exp_name, snapshot_name + 'pth'))
        torch.save(optimizer.state_dict(), os.path.join(
            ckpt_path, exp_name, 'opt_' + snapshot_name + 'pth'))

        with open(exp_name + '.txt', 'a') as f:
            f.write(snapshot_name + '\n')

        to_save_dir = os.path.join(ckpt_path, exp_name, str(epoch + 1))
        rmrf_mkdir(to_save_dir)

        x = []
        for idx, tensor in enumerate(zip(input_batches, prediction_batches, label_batches)):
            if random.random() > val_args['tensorboard_img_sample_rate']:
                continue
            pil_input = restore(tensor[0])
            pil_output = colorize_mask(tensor[1])
            pil_label = colorize_mask(tensor[2])
            pil_input.save(os.path.join(to_save_dir, '%d_img.png' % idx))
            pil_output.save(os.path.join(to_save_dir, '%d_out.png' % idx))
            pil_label.save(os.path.join(to_save_dir, '%d_label.png' % idx))
            x.extend([pil_to_tensor(pil_input.convert('RGB')), pil_to_tensor(pil_label.convert('RGB')),
                      pil_to_tensor(pil_output.convert('RGB'))])
        x = torch.stack(x, 0)
        x = vutils.make_grid(x, nrow=3, padding=5)
        writer.add_image(snapshot_name, x)

    print '--------------------------------------------------------'
    print '[val loss %.4f], [mean iu %.4f]' % (val_loss, mean_iu)
    print '[best val loss %.4f], [mean iu %.4f], [epoch %d]' % (
        train_record['best_val_loss'], train_record['corr_mean_iu'], train_record['corr_epoch'])
    print '--------------------------------------------------------'

    net.train()
    criterion.cuda()


if __name__ == '__main__':
    main()
