import os
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
    weight[list(range(in_channels)), list(range(out_channels)), :, :] = filt
    return torch.from_numpy(weight).float()


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)


class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, weight=None, size_average=True, ignore_index=255):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss((1 - F.softmax(inputs)) ** self.gamma * F.log_softmax(inputs), targets)


def _fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist


def evaluate(predictions, gts, num_classes):
    hist = np.zeros((num_classes, num_classes))
    for lp, lt in zip(predictions, gts):
        hist += _fast_hist(lp.flatten(), lt.flatten(), num_classes)
    # axis 0: gt, axis 1: prediction
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class PolyLR(object):
    def __init__(self, optimizer, curr_iter, max_iter, lr_decay):
        self.max_iter = float(max_iter)
        self.init_lr_groups = []
        for p in optimizer.param_groups:
            self.init_lr_groups.append(p['lr'])
        self.param_groups = optimizer.param_groups
        self.curr_iter = curr_iter
        self.lr_decay = lr_decay

    def step(self):
        for idx, p in enumerate(self.param_groups):
            p['lr'] = self.init_lr_groups[idx] * (1 - self.curr_iter / self.max_iter) ** self.lr_decay


# just a try, not recommend to use
class Conv2dDeformable(nn.Module):
    def __init__(self, regular_filter, cuda=True):
        super(Conv2dDeformable, self).__init__()
        assert isinstance(regular_filter, nn.Conv2d)
        self.regular_filter = regular_filter
        self.offset_filter = nn.Conv2d(regular_filter.in_channels, 2 * regular_filter.in_channels, kernel_size=3,
                                       padding=1, bias=False)
        self.offset_filter.weight.data.normal_(0, 0.0005)
        self.input_shape = None
        self.grid_w = None
        self.grid_h = None
        self.cuda = cuda

    def forward(self, x):
        x_shape = x.size()  # (b, c, h, w)
        offset = self.offset_filter(x)  # (b, 2*c, h, w)
        offset_w, offset_h = torch.split(offset, self.regular_filter.in_channels, 1)  # (b, c, h, w)
        offset_w = offset_w.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, h, w)
        offset_h = offset_h.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, h, w)
        if not self.input_shape or self.input_shape != x_shape:
            self.input_shape = x_shape
            grid_w, grid_h = np.meshgrid(np.linspace(-1, 1, x_shape[3]), np.linspace(-1, 1, x_shape[2]))  # (h, w)
            grid_w = torch.Tensor(grid_w)
            grid_h = torch.Tensor(grid_h)
            if self.cuda:
                grid_w = grid_w.cuda()
                grid_h = grid_h.cuda()
            self.grid_w = nn.Parameter(grid_w)
            self.grid_h = nn.Parameter(grid_h)
        offset_w = offset_w + self.grid_w  # (b*c, h, w)
        offset_h = offset_h + self.grid_h  # (b*c, h, w)
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3])).unsqueeze(1)  # (b*c, 1, h, w)
        x = F.grid_sample(x, torch.stack((offset_h, offset_w), 3))  # (b*c, h, w)
        x = x.contiguous().view(-1, int(x_shape[1]), int(x_shape[2]), int(x_shape[3]))  # (b, c, h, w)
        x = self.regular_filter(x)
        return x


def sliced_forward(single_forward):
    def _pad(x, crop_size):
        h, w = x.size()[2:]
        pad_h = max(crop_size - h, 0)
        pad_w = max(crop_size - w, 0)
        x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, pad_h, pad_w

    def wrapper(self, x):
        batch_size, _, ori_h, ori_w = x.size()
        if self.training and self.use_aux:
            outputs_all_scales = Variable(torch.zeros((batch_size, self.num_classes, ori_h, ori_w))).cuda()
            aux_all_scales = Variable(torch.zeros((batch_size, self.num_classes, ori_h, ori_w))).cuda()
            for s in self.scales:
                new_size = (int(ori_h * s), int(ori_w * s))
                scaled_x = F.upsample(x, size=new_size, mode='bilinear')
                scaled_x = Variable(scaled_x).cuda()
                scaled_h, scaled_w = scaled_x.size()[2:]
                long_size = max(scaled_h, scaled_w)
                print(scaled_x.size())

                if long_size > self.crop_size:
                    count = torch.zeros((scaled_h, scaled_w))
                    outputs = Variable(torch.zeros((batch_size, self.num_classes, scaled_h, scaled_w))).cuda()
                    aux_outputs = Variable(torch.zeros((batch_size, self.num_classes, scaled_h, scaled_w))).cuda()
                    stride = int(ceil(self.crop_size * self.stride_rate))
                    h_step_num = int(ceil((scaled_h - self.crop_size) / stride)) + 1
                    w_step_num = int(ceil((scaled_w - self.crop_size) / stride)) + 1
                    for yy in range(h_step_num):
                        for xx in range(w_step_num):
                            sy, sx = yy * stride, xx * stride
                            ey, ex = sy + self.crop_size, sx + self.crop_size
                            x_sub = scaled_x[:, :, sy: ey, sx: ex]
                            x_sub, pad_h, pad_w = _pad(x_sub, self.crop_size)
                            print(x_sub.size())
                            outputs_sub, aux_sub = single_forward(self, x_sub)

                            if sy + self.crop_size > scaled_h:
                                outputs_sub = outputs_sub[:, :, : -pad_h, :]
                                aux_sub = aux_sub[:, :, : -pad_h, :]

                            if sx + self.crop_size > scaled_w:
                                outputs_sub = outputs_sub[:, :, :, : -pad_w]
                                aux_sub = aux_sub[:, :, :, : -pad_w]

                            outputs[:, :, sy: ey, sx: ex] = outputs_sub
                            aux_outputs[:, :, sy: ey, sx: ex] = aux_sub

                            count[sy: ey, sx: ex] += 1
                    count = Variable(count).cuda()
                    outputs = (outputs / count)
                    aux_outputs = (outputs / count)
                else:
                    scaled_x, pad_h, pad_w = _pad(scaled_x, self.crop_size)
                    outputs, aux_outputs = single_forward(self, scaled_x)
                    outputs = outputs[:, :, : -pad_h, : -pad_w]
                    aux_outputs = aux_outputs[:, :, : -pad_h, : -pad_w]
                outputs_all_scales += outputs
                aux_all_scales += aux_outputs
            return outputs_all_scales / len(self.scales), aux_all_scales
        else:
            outputs_all_scales = Variable(torch.zeros((batch_size, self.num_classes, ori_h, ori_w))).cuda()
            for s in self.scales:
                new_size = (int(ori_h * s), int(ori_w * s))
                scaled_x = F.upsample(x, size=new_size, mode='bilinear')
                scaled_h, scaled_w = scaled_x.size()[2:]
                long_size = max(scaled_h, scaled_w)

                if long_size > self.crop_size:
                    count = torch.zeros((scaled_h, scaled_w))
                    outputs = Variable(torch.zeros((batch_size, self.num_classes, scaled_h, scaled_w))).cuda()
                    stride = int(ceil(self.crop_size * self.stride_rate))
                    h_step_num = int(ceil((scaled_h - self.crop_size) / stride)) + 1
                    w_step_num = int(ceil((scaled_w - self.crop_size) / stride)) + 1
                    for yy in range(h_step_num):
                        for xx in range(w_step_num):
                            sy, sx = yy * stride, xx * stride
                            ey, ex = sy + self.crop_size, sx + self.crop_size
                            x_sub = scaled_x[:, :, sy: ey, sx: ex]
                            x_sub, pad_h, pad_w = _pad(x_sub, self.crop_size)

                            outputs_sub = single_forward(self, x_sub)

                            if sy + self.crop_size > scaled_h:
                                outputs_sub = outputs_sub[:, :, : -pad_h, :]

                            if sx + self.crop_size > scaled_w:
                                outputs_sub = outputs_sub[:, :, :, : -pad_w]

                            outputs[:, :, sy: ey, sx: ex] = outputs_sub

                            count[sy: ey, sx: ex] += 1
                    count = Variable(count).cuda()
                    outputs = (outputs / count)
                else:
                    scaled_x, pad_h, pad_w = _pad(scaled_x, self.crop_size)
                    outputs = single_forward(self, scaled_x)
                    outputs = outputs[:, :, : -pad_h, : -pad_w]
                outputs_all_scales += outputs
            return outputs_all_scales

    return wrapper
