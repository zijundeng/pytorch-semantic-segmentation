import numbers
import random

import numpy as np
import torch
from PIL import Image, ImageOps, ImageFilter


class RandomVerticalFlip(object):
    def __call__(self, img):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


class RandomGaussianBlur(object):
    def __call__(self, img):
        if random.random() < 0.2:
            return img.filter(ImageFilter.BLUR)
        return img


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


class SimultaneousCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img1, img2):
        for t in self.transforms:
            img1, img2 = t(img1, img2)
        return img1, img2


class SimultaneousRandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img1, img2):
        if self.padding > 0:
            img1 = ImageOps.expand(img1, border=self.padding, fill=0)
            img2 = ImageOps.expand(img2, border=self.padding, fill=0)

        assert img1.size == img2.size
        w, h = img1.size
        th, tw = self.size
        if w == tw and h == th:
            return img1, img2
        if w < tw or h < th:
            return img1.resize((tw, th), Image.BILINEAR), img2.resize((tw, th), Image.NEAREST)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img1.crop((x1, y1, x1 + tw, y1 + th)), img2.crop((x1, y1, x1 + tw, y1 + th))


class SimultaneousRandomHorizontallyFlip(object):
    def __call__(self, img1, img2):
        if random.random() < 0.5:
            return img1.transpose(Image.FLIP_LEFT_RIGHT), img2.transpose(Image.FLIP_LEFT_RIGHT)
        return img1, img2


class SimultaneousRandomScale(object):
    def __init__(self, scale_range, interpolation=Image.NEAREST):
        self.lb = scale_range[0]
        self.ub = scale_range[1]
        assert self.lb < self.ub
        self.interpolation = interpolation

    def __call__(self, img1, img2):
        assert img1.size == img2.size
        w, h = img1.size
        ow = int(((self.ub - self.lb) * random.random() + self.lb) * w)
        oh = int(((self.ub - self.lb) * random.random() + self.lb) * h)
        return img1.resize((ow, oh), self.interpolation), img2.resize((ow, oh), self.interpolation)


class SimultaneousFreeScale(object):
    def __init__(self, size, interpolation=Image.NEAREST):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img1, img2):
        return img1.resize((self.size[0], self.size[1]), self.interpolation), img2.resize((self.size[1], self.size[0]),
                                                                                          self.interpolation)


class SimultaneousScale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img1, img2):
        assert img1.size == img2.size
        w, h = img1.size
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img1, img2
        if w < h:
            ow = self.size
            oh = int(self.size * h / w)
            return img1.resize((ow, oh), Image.BILINEAR), img2.resize((ow, oh), Image.NEAREST)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img1.resize((ow, oh), Image.BILINEAR), img2.resize((ow, oh), Image.NEAREST)


class CRF(object):
    def __init__(self):
        pass

    def __call__(self, img):
        pass
