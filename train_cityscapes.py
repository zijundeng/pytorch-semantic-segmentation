import os

import torch
import torchvision.transforms as standard_transforms
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
from models import PSPNet
from utils.io import rmrf_mkdir
from utils.loss import CrossEntropyLoss2dOld
from utils.training import calculate_mean_iu

cudnn.benchmark = True


def main():
    training_batch_size = 3
    validation_batch_size = 1
    epoch_num = 800
    iter_freq_print_training_log = 10
    new_lr = 1e-5
    pretrained_lr = 1e-5

    # net = PSPNet(pretrained=True, num_classes=num_classes, input_size=(384, 768)).cuda()
    # curr_epoch = 0

    net = PSPNet(pretrained=False, num_classes=num_classes, input_size=(384, 768)).cuda()
    snapshot = 'epoch_19_validation_loss_69231040.0000_mean_iu_0.3243_lr_0.00001000.pth'
    net.load_state_dict(torch.load(os.path.join(ckpt_path, snapshot)))
    split_res = snapshot.split('_')
    curr_epoch = int(split_res[1])

    net.train()

    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_simul_transform = simul_transforms.Compose([
        simul_transforms.Scale(439),
        simul_transforms.RandomCrop((384, 768)),
        simul_transforms.RandomHorizontallyFlip()
    ])
    train_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    val_simul_transform = simul_transforms.Compose([
        simul_transforms.Scale(439),
        simul_transforms.CenterCrop((384, 768)),
    ])
    val_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    restore_transform = standard_transforms.Compose([
        expanded_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])

    train_set = CityScapes('train', simul_transform=train_simul_transform, transform=train_transform,
                           target_transform=expanded_transforms.MaskToTensor())
    train_loader = DataLoader(train_set, batch_size=training_batch_size, num_workers=16, shuffle=True)
    val_set = CityScapes('val', simul_transform=val_simul_transform, transform=val_transform,
                         target_transform=expanded_transforms.MaskToTensor())
    val_loader = DataLoader(val_set, batch_size=validation_batch_size, num_workers=16, shuffle=False)

    criterion = CrossEntropyLoss2dOld(ignored_label=ignored_label)
    optimizer = optim.Adam([
        {'params': [param for name, param in net.named_parameters() if
                    name[-4:] == 'bias' and ('ppm' in name or 'final' in name)], 'lr': new_lr},
        {'params': [param for name, param in net.named_parameters() if
                    name[-4:] != 'bias' and ('ppm' in name or 'final' in name)],
         'lr': new_lr, 'weight_decay': 5e-4},
        {'params': [param for name, param in net.named_parameters() if
                    name[-4:] == 'bias' and not ('ppm' in name or 'final' in name)], 'lr': pretrained_lr},
        {'params': [param for name, param in net.named_parameters() if
                    name[-4:] != 'bias' and not ('ppm' in name or 'final' in name)], 'lr': pretrained_lr,
         'weight_decay': 5e-4}
    ])

    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)

    best = [1e20, -1, -1]  # [best_val_loss, best_mean_iu, best_epoch]

    for epoch in range(curr_epoch, epoch_num):
        train(train_loader, net, criterion, optimizer, epoch, iter_freq_print_training_log)
        validate(epoch, val_loader, net, criterion, restore_transform, best, new_lr)


def train(train_loader, net, criterion, optimizer, epoch, iter_freq_print_training_log):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % iter_freq_print_training_log == 0:
            prediction = outputs.data.max(1)[1].squeeze_(1).cpu().numpy()
            mean_iu = calculate_mean_iu(prediction, labels.data.cpu().numpy(), num_classes)
            print '[epoch %d], [iter %d], [training batch loss %.4f], [mean_iu %.4f]' % (
                epoch + 1, i + 1, loss.data[0], mean_iu)


def validate(epoch, val_loader, net, criterion, restore, best, lr):
    net.eval()
    batch_inputs = []
    batch_outputs = []
    batch_labels = []
    for vi, data in enumerate(val_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs, volatile=True).cuda()
        labels = Variable(labels, volatile=True).cuda()

        outputs = net(inputs)

        batch_inputs.append(inputs.cpu())
        batch_outputs.append(outputs.cpu())
        batch_labels.append(labels.cpu())

        if vi > 100:
            break

    batch_inputs = torch.cat(batch_inputs)
    batch_outputs = torch.cat(batch_outputs)
    batch_labels = torch.cat(batch_labels)
    val_loss = criterion(batch_outputs, batch_labels)
    val_loss = val_loss.data[0]

    batch_inputs = batch_inputs.data
    batch_outputs = batch_outputs.data
    batch_labels = batch_labels.data.numpy()
    batch_prediction = batch_outputs.max(1)[1].squeeze_(1).numpy()

    mean_iu = calculate_mean_iu(batch_prediction, batch_labels, num_classes)

    if val_loss < best[0]:
        best[0] = val_loss
        best[1] = mean_iu
        best[2] = epoch
        torch.save(net.state_dict(), os.path.join(
            ckpt_path, 'epoch_%d_validation_loss_%.4f_mean_iu_%.4f_lr_%.8f.pth' % (epoch + 1, val_loss, mean_iu, lr)))

        with open('log.txt', 'a') as f:
            f.write('epoch_%d_validation_loss_%.4f_mean_iu_%.4f_lr_%.8f\n' % (epoch + 1, val_loss, mean_iu, lr))

        to_save_dir = os.path.join(ckpt_path, str(epoch + 1))
        rmrf_mkdir(to_save_dir)

        for idx, tensor in enumerate(zip(batch_inputs, batch_prediction, batch_labels)):
            pil_input = restore(tensor[0])
            pil_output = colorize_mask(tensor[1])
            pil_label = colorize_mask(tensor[2])
            pil_input.save(os.path.join(to_save_dir, '%d_img.png' % idx))
            pil_output.save(os.path.join(to_save_dir, '%d_out.png' % idx))
            pil_label.save(os.path.join(to_save_dir, '%d_label.png' % idx))

    print '--------------------------------------------------------'
    print '[validation loss %.4f]' % val_loss
    print '[best validation loss %.4f], [best_mean_iu %.4f], [best epoch %d]' % (
        best[0], best[1], best[2] + 1)
    print '--------------------------------------------------------'

    net.train()


if __name__ == '__main__':
    main()
