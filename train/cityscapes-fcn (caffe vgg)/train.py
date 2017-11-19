import datetime
import os
import random

import numpy as np
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from tensorboard import SummaryWriter
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import utils.joint_transforms as joint_transforms
import utils.transforms as extended_transforms
from datasets import cityscapes
from models import *
from utils import check_mkdir, evaluate, AverageMeter, CrossEntropyLoss2d

cudnn.benchmark = True

ckpt_path = '../../ckpt'
exp_name = 'cityscapes-fcn8s (caffe vgg)'
writer = SummaryWriter(os.path.join(ckpt_path, 'exp', exp_name))

args = {
    'train_batch_size': 12,
    'epoch_num': 500,
    'lr': 1e-10,
    'weight_decay': 5e-4,
    'input_size': (256, 512),
    'momentum': 0.99,
    'lr_patience': 100,  # large patience denotes fixed lr
    'snapshot': '',  # empty string denotes no snapshot
    'print_freq': 20,
    'val_batch_size': 16,
    'val_save_to_img_file': False,
    'val_img_sample_rate': 0.05  # randomly sample some validation results to display
}


def main():
    net = FCN8s(num_classes=cityscapes.num_classes, caffe=True).cuda()

    if len(args['snapshot']) == 0:
        curr_epoch = 1
        args['best_record'] = {'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}
    else:
        print('training resumes from ' + args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'])))
        split_snapshot = args['snapshot'].split('_')
        curr_epoch = int(split_snapshot[1]) + 1
        args['best_record'] = {'epoch': int(split_snapshot[1]), 'val_loss': float(split_snapshot[3]),
                               'acc': float(split_snapshot[5]), 'acc_cls': float(split_snapshot[7]),
                               'mean_iu': float(split_snapshot[9]), 'fwavacc': float(split_snapshot[11])}

    net.train()

    mean_std = ([103.939, 116.779, 123.68], [1.0, 1.0, 1.0])

    short_size = int(min(args['input_size']) / 0.875)
    train_joint_transform = joint_transforms.Compose([
        joint_transforms.Scale(short_size),
        joint_transforms.RandomCrop(args['input_size']),
        joint_transforms.RandomHorizontallyFlip()
    ])
    val_joint_transform = joint_transforms.Compose([
        joint_transforms.Scale(short_size),
        joint_transforms.CenterCrop(args['input_size'])
    ])
    input_transform = standard_transforms.Compose([
        extended_transforms.FlipChannels(),
        standard_transforms.ToTensor(),
        standard_transforms.Lambda(lambda x: x.mul_(255)),
        standard_transforms.Normalize(*mean_std)
    ])
    target_transform = extended_transforms.MaskToTensor()
    restore_transform = standard_transforms.Compose([
        extended_transforms.DeNormalize(*mean_std),
        standard_transforms.Lambda(lambda x: x.div_(255)),
        standard_transforms.ToPILImage(),
        extended_transforms.FlipChannels()
    ])
    visualize = standard_transforms.ToTensor()

    train_set = cityscapes.CityScapes('fine', 'train', joint_transform=train_joint_transform,
                                      transform=input_transform, target_transform=target_transform)
    train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=8, shuffle=True)
    val_set = cityscapes.CityScapes('fine', 'val', joint_transform=val_joint_transform, transform=input_transform,
                                    target_transform=target_transform)
    val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], num_workers=8, shuffle=False)

    criterion = CrossEntropyLoss2d(size_average=False, ignore_index=cityscapes.ignore_label).cuda()

    optimizer = optim.Adam([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], betas=(args['momentum'], 0.999))

    if len(args['snapshot']) > 0:
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, 'opt_' + args['snapshot'])))
        optimizer.param_groups[0]['lr'] = 2 * args['lr']
        optimizer.param_groups[1]['lr'] = args['lr']

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt'), 'w').write(str(args) + '\n\n')

    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=args['lr_patience'], min_lr=1e-10, verbose=True)
    for epoch in range(curr_epoch, args['epoch_num'] + 1):
        train(train_loader, net, criterion, optimizer, epoch, args)
        val_loss = validate(val_loader, net, criterion, optimizer, epoch, args, restore_transform, visualize)
        scheduler.step(val_loss)


def train(train_loader, net, criterion, optimizer, epoch, train_args):
    train_loss = AverageMeter()
    curr_iter = (epoch - 1) * len(train_loader)
    for i, data in enumerate(train_loader):
        inputs, labels = data
        assert inputs.size()[2:] == labels.size()[1:]
        N = inputs.size(0)
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()

        optimizer.zero_grad()
        outputs = net(inputs)
        assert outputs.size()[2:] == labels.size()[1:]
        assert outputs.size()[1] == cityscapes.num_classes

        loss = criterion(outputs, labels) / N
        loss.backward()
        optimizer.step()

        train_loss.update(loss.data[0], N)

        curr_iter += 1
        writer.add_scalar('train_loss', train_loss.avg, curr_iter)

        if (i + 1) % train_args['print_freq'] == 0:
            print('[epoch %d], [iter %d / %d], [train loss %.5f]' % (
                epoch, i + 1, len(train_loader), train_loss.avg))


def validate(val_loader, net, criterion, optimizer, epoch, train_args, restore, visualize):
    net.eval()

    val_loss = AverageMeter()
    inputs_all, gts_all, predictions_all = [], [], []

    for vi, data in enumerate(val_loader):
        inputs, gts = data
        N = inputs.size(0)
        inputs = Variable(inputs, volatile=True).cuda()
        gts = Variable(gts, volatile=True).cuda()

        outputs = net(inputs)
        predictions = outputs.data.max(1)[1].squeeze_(1).cpu().numpy()

        val_loss.update(criterion(outputs, gts).data[0] / N, N)

        for i in inputs:
            if random.random() > train_args['val_img_sample_rate']:
                inputs_all.append(None)
            else:
                inputs_all.append(i.data.cpu())
        gts_all.append(gts.data.cpu().numpy())
        predictions_all.append(predictions)

    gts_all = np.concatenate(gts_all)
    predictions_all = np.concatenate(predictions_all)

    acc, acc_cls, mean_iu, fwavacc = evaluate(predictions_all, gts_all, cityscapes.num_classes)

    if mean_iu > train_args['best_record']['mean_iu']:
        train_args['best_record']['val_loss'] = val_loss.avg
        train_args['best_record']['epoch'] = epoch
        train_args['best_record']['acc'] = acc
        train_args['best_record']['acc_cls'] = acc_cls
        train_args['best_record']['mean_iu'] = mean_iu
        train_args['best_record']['fwavacc'] = fwavacc
        snapshot_name = 'epoch_%d_loss_%.5f_acc_%.5f_acc-cls_%.5f_mean-iu_%.5f_fwavacc_%.5f_lr_%.10f' % (
            epoch, val_loss.avg, acc, acc_cls, mean_iu, fwavacc, optimizer.param_groups[1]['lr']
        )
        torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, snapshot_name + '.pth'))
        torch.save(optimizer.state_dict(), os.path.join(ckpt_path, exp_name, 'opt_' + snapshot_name + '.pth'))

        if train_args['val_save_to_img_file']:
            to_save_dir = os.path.join(ckpt_path, exp_name, str(epoch))
            check_mkdir(to_save_dir)

        val_visual = []
        for idx, data in enumerate(zip(inputs_all, gts_all, predictions_all)):
            if data[0] is None:
                continue
            input_pil = restore(data[0])
            gt_pil = cityscapes.colorize_mask(data[1])
            predictions_pil = cityscapes.colorize_mask(data[2])
            if train_args['val_save_to_img_file']:
                input_pil.save(os.path.join(to_save_dir, '%d_input.png' % idx))
                predictions_pil.save(os.path.join(to_save_dir, '%d_prediction.png' % idx))
                gt_pil.save(os.path.join(to_save_dir, '%d_gt.png' % idx))
            val_visual.extend([visualize(input_pil.convert('RGB')), visualize(gt_pil.convert('RGB')),
                               visualize(predictions_pil.convert('RGB'))])
        val_visual = torch.stack(val_visual, 0)
        val_visual = vutils.make_grid(val_visual, nrow=3, padding=5)
        writer.add_image(snapshot_name, val_visual)

        print('-----------------------------------------------------------------------------------------------------------')
    print('[epoch %d], [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f]' % (
        epoch, val_loss.avg, acc, acc_cls, mean_iu, fwavacc))

    print('best record: [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f], [epoch %d]' % (
        train_args['best_record']['val_loss'], train_args['best_record']['acc'], train_args['best_record']['acc_cls'],
        train_args['best_record']['mean_iu'], train_args['best_record']['fwavacc'], train_args['best_record']['epoch']))

    print('-----------------------------------------------------------------------------------------------------------')

    writer.add_scalar('val_loss', val_loss.avg, epoch)
    writer.add_scalar('acc', acc, epoch)
    writer.add_scalar('acc_cls', acc_cls, epoch)
    writer.add_scalar('mean_iu', mean_iu, epoch)
    writer.add_scalar('fwavacc', fwavacc, epoch)
    writer.add_scalar('lr', optimizer.param_groups[1]['lr'], epoch)

    net.train()
    return val_loss.avg


if __name__ == '__main__':
    main()
