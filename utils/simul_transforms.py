import numbers
import random

from PIL import Image, ImageOps


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img1, img2):
        for t in self.transforms:
            img1, img2 = t(img1, img2)
        return img1, img2


class RandomCrop(object):
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
        th, tw  = self.size
        if w == tw and h == th:
            return img1, img2
        if w < tw or h < th:
            return img1.resize((tw, th), Image.BILINEAR), img2.resize((tw, th), Image.NEAREST)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img1.crop((x1, y1, x1 + tw, y1 + th)), img2.crop((x1, y1, x1 + tw, y1 + th))


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img1, img2):
        w, h = img1.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img1.crop((x1, y1, x1 + tw, y1 + th)), img2.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontallyFlip(object):
    def __call__(self, img1, img2):
        if random.random() < 0.5:
            return img1.transpose(Image.FLIP_LEFT_RIGHT), img2.transpose(Image.FLIP_LEFT_RIGHT)
        return img1, img2


class FreeScale(object):
    def __init__(self, size, interpolation=Image.NEAREST):
        self.size = size  # (h, w)
        self.interpolation = interpolation

    def __call__(self, img1, img2):
        return img1.resize((self.size[1], self.size[0]), self.interpolation), img2.resize(self.size, self.interpolation)


class Scale(object):
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
