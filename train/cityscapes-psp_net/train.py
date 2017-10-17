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
from torch.utils.data import DataLoader

import utils.joint_transforms as joint_transforms
import utils.transforms as extended_transforms
from datasets import cityscapes
from models import *
from utils import check_mkdir, evaluate, AverageMeter, CrossEntropyLoss2d

cudnn.benchmark = True

ckpt_path = '../../ckpt'
exp_name = 'cityscapes (fine)-psp_net'
writer = SummaryWriter(os.path.join(ckpt_path, 'exp', exp_name))

args = {
    'train_batch_size': 1,
    'lr': 1e-2,
    'lr_decay': 0.9,
    'max_iter': 9e4,
    'crop_size': 713,
    'stride_rate': 2 / 3.,
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'snapshot': '',  # empty string denotes learning from scratch
    'print_freq': 10,
    'val_save_to_img_file': False,
    'val_img_sample_rate': 0.01,  # randomly sample some validation results to display,
    'val_img_display_size': 256,
}


def main():
    net = PSPNet(num_classes=cityscapes.num_classes).cuda()

    if len(args['snapshot']) == 0:
        # net.load_state_dict(torch.load(os.path.join(ckpt_path, 'cityscapes (coarse)-psp_net', 'xx.pth')))
        curr_epoch = 1
        args['best_record'] = {'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}
    else:
        print 'training resumes from ' + args['snapshot']
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'])))
        split_snapshot = args['snapshot'].split('_')
        curr_epoch = int(split_snapshot[1]) + 1
        args['best_record'] = {'epoch': int(split_snapshot[1]), 'val_loss': float(split_snapshot[3]),
                               'acc': float(split_snapshot[5]), 'acc_cls': float(split_snapshot[7]),
                               'mean_iu': float(split_snapshot[9]), 'fwavacc': float(split_snapshot[11])}

    net.train()

    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    train_joint_transform = joint_transforms.Compose([
        joint_transforms.RandomRotate(10),
        joint_transforms.RandomHorizontallyFlip(),
        joint_transforms.SlidingCrop(args['crop_size'], args['stride_rate'], cityscapes.ignore_label)
    ])
    train_input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    val_input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    target_transform = extended_transforms.MaskToTensor()
    restore_transform = standard_transforms.Compose([
        extended_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage(),
    ])
    visualize = standard_transforms.Compose([
        standard_transforms.Scale(args['val_img_display_size']),
        standard_transforms.ToTensor()
    ])

    train_set = cityscapes.CityScapes('fine', 'train', joint_transform=train_joint_transform,
                                      transform=train_input_transform, target_transform=target_transform)
    train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=8, shuffle=True)
    val_set = cityscapes.CityScapes('fine', 'val', transform=val_input_transform, target_transform=target_transform)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=8, shuffle=True)

    criterion = CrossEntropyLoss2d(size_average=True, ignore_index=cityscapes.ignore_label).cuda()

    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'])

    if len(args['snapshot']) > 0:
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, 'opt_' + args['snapshot'])))
        optimizer.param_groups[0]['lr'] = 2 * args['lr']
        optimizer.param_groups[1]['lr'] = args['lr']

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt'), 'w').write(str(args) + '\n\n')

    train(train_loader, net, criterion, optimizer, curr_epoch, args, val_loader, restore_transform, visualize)


def train(train_loader, net, criterion, optimizer, curr_epoch, train_args, val_loader, restore, visualize):
    while True:
        train_main_loss = AverageMeter()
        train_aux_loss = AverageMeter()
        curr_iter = (curr_epoch - 1) * len(train_loader)
        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 * train_args['lr'] * (1 - float(curr_iter) / train_args['max_iter']
                                                                      ) ** train_args['lr_decay']
            optimizer.param_groups[1]['lr'] = train_args['lr'] * (1 - float(curr_iter) / train_args['max_iter']
                                                                  ) ** train_args['lr_decay']

            inputs, labels = data
            if len(inputs.size()) == 5 and len(labels.size()) == 4:
                inputs.transpose_(0, 1)
                labels.transpose_(0, 1)
            else:
                inputs.unsqueeze_(0)
                labels.unsqueeze_(0)

            assert inputs.size()[3:] == labels.size()[2:]
            N = inputs.size(1) * inputs.size(3) * inputs.size(4)

            for inputs_slice, labels_slice in zip(inputs, labels):
                inputs_slice = Variable(inputs_slice).cuda()
                labels_slice = Variable(labels_slice).cuda()

                optimizer.zero_grad()
                outputs, aux = net(inputs_slice)
                assert outputs.size()[2:] == labels_slice.size()[1:]
                assert outputs.size()[1] == cityscapes.num_classes

                main_loss = criterion(outputs, labels_slice)
                aux_loss = criterion(aux, labels_slice)
                loss = main_loss + 0.4 * aux_loss
                loss.backward()
                optimizer.step()

                train_main_loss.update(main_loss.data[0], N)
                train_aux_loss.update(aux_loss.data[0], N)

            curr_iter += 1
            writer.add_scalar('train_main_loss', train_main_loss.avg, curr_iter)
            writer.add_scalar('train_aux_loss', train_aux_loss.avg, curr_iter)
            writer.add_scalar('lr', optimizer.param_groups[1]['lr'], curr_iter)

            if (i + 1) % train_args['print_freq'] == 0:
                print '[epoch %d], [iter %d / %d], [train main loss %.5f], [train aux loss %.5f]. [lr %.10f]' % (
                    curr_epoch, i + 1, len(train_loader), train_main_loss.avg, train_aux_loss.avg,
                    optimizer.param_groups[1]['lr'])
            if curr_iter >= train_args['max_iter']:
                return
        validate(val_loader, net, criterion, optimizer, curr_epoch, train_args, restore, visualize)
        curr_epoch += 1


def validate(val_loader, net, criterion, optimizer, epoch, train_args, restore, visualize):
    net.eval()

    val_loss = AverageMeter()
    inputs_all = []
    gts_all = np.zeros((len(val_loader), 1024, 2048), dtype=int)
    predictions_all = np.zeros((len(val_loader), 1024, 2048), dtype=int)
    for vi, data in enumerate(val_loader):
        inputs, gts = data
        N = inputs.size(0) * inputs.size(2) * inputs.size(3)
        inputs = Variable(inputs, volatile=True).cuda()
        gts = Variable(gts, volatile=True).cuda()

        outputs = net(inputs)
        predictions_all[vi, :, :] = outputs.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()

        val_loss.update(criterion(outputs, gts).data[0], N)

        for i in inputs:
            if random.random() < train_args['val_img_sample_rate'] or len(inputs_all) == 0:
                inputs_all.append(i.data.cpu())
            else:
                inputs_all.append(None)
        gts_all[vi, :, :] = gts.data.squeeze_(0).cpu().numpy()

        print 'validating: %d / %d' % (vi + 1, len(val_loader))

    acc, acc_cls, mean_iu, fwavacc = evaluate(predictions_all, gts_all, cityscapes.num_classes)

    if mean_iu > train_args['best_record']['mean_iu']:
        train_args['best_record']['val_loss'] = val_loss.avg
        train_args['best_record']['epoch'] = epoch
        train_args['best_record']['acc'] = acc
        train_args['best_record']['acc_cls'] = acc_cls
        train_args['best_record']['mean_iu'] = mean_iu
        train_args['best_record']['fwavacc'] = fwavacc
        snapshot_name = 'epoch_%d_loss_%.5f_acc_%.5f_acc-cls_%.5f_mean-iu_%.5f_fwavacc_%.5f_lr_%.10f' % (
            epoch, val_loss.avg, acc, acc_cls, mean_iu, fwavacc, optimizer.param_groups[1]['lr'])
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

    print '-----------------------------------------------------------------------------------------------------------'
    print '[epoch %d], [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f]' % (
        epoch, val_loss.avg, acc, acc_cls, mean_iu, fwavacc)

    print 'best record: [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f], [epoch %d]' % (
        train_args['best_record']['val_loss'], train_args['best_record']['acc'], train_args['best_record']['acc_cls'],
        train_args['best_record']['mean_iu'], train_args['best_record']['fwavacc'], train_args['best_record']['epoch'])

    print '-----------------------------------------------------------------------------------------------------------'

    writer.add_scalar('val_loss', val_loss.avg, epoch)
    writer.add_scalar('acc', acc, epoch)
    writer.add_scalar('acc_cls', acc_cls, epoch)
    writer.add_scalar('mean_iu', mean_iu, epoch)
    writer.add_scalar('fwavacc', fwavacc, epoch)

    net.train()
    return val_loss.avg


if __name__ == '__main__':
    main()
