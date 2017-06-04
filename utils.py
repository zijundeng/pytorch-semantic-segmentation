import random
from PIL import Image

import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss, _assert_no_grad


def crf(img_batch):
    pass


class CrossEntropyLoss2d(_WeightedLoss):
    def forward(self, input, target, ignore_label=255):
        _assert_no_grad(target)
        n, c, h, w = input.size()
        input = input.transpose(1, 2).transpose(2, 3).contiguous()
        input = input[target.view(n, h, w, 1).repeat(1, 1, 1, c) != ignore_label].view(-1, c)
        target = target[target != ignore_label].view(-1)
        return F.cross_entropy(input, target, self.weight, self.size_average)


class RandomVerticalFlip(object):
    """Randomly vertically flips the given PIL.Image with a probability of 0.5
    """

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
