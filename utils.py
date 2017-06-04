import random
from PIL import Image
from torch import nn
import math
import collections

import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss, _assert_no_grad


def crf(img_batch):
    pass


class CrossEntropyLoss2d(_WeightedLoss):
    def forward(self, inputs, targets, ignored_label=None):
        _assert_no_grad(targets)
        n, c, h, w = inputs.size()
        inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous()

        if ignored_label is None:
            inputs = inputs.view(-1, c)
            targets = targets.view(-1)
        else:
            inputs = inputs[targets.view(n, h, w, 1).repeat(1, 1, 1, c) != ignored_label].view(-1, c)
            targets = targets[targets != ignored_label].view(-1)

        return F.cross_entropy(inputs, targets, self.weight, self.size_average)


class RandomVerticalFlip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2. / n))
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                n = module.weight.size(1)
                module.weight.data.normal_(0, math.sqrt(2. / n))
                module.bias.data.zero_()