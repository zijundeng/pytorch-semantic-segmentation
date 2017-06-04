import os
from PIL import Image

import numpy as np
import torch
from torch.utils import data


def default_loader(path):
    return Image.open(path).convert('RGB')


def mask_loader(path):
    return Image.open(path)


def make_dataset(root):
    mask_dir_path = os.path.join(root, 'SegmentationClassAug')
    image_dir_path = os.path.join(root, 'JPEGImages')
    images = []
    for img_name in [os.path.splitext(img_name)[0] for img_name in os.listdir(mask_dir_path)]:
        item = (os.path.join(image_dir_path, img_name + '.jpg'), os.path.join(mask_dir_path, img_name + '.png'))
        images.append(item)
    return images


class VOC(data.Dataset):
    def __init__(self, root, transform=None):
        self.imgs = make_dataset(root)
        if len(self.imgs) == 0:
            raise (RuntimeError('Found 0 images, please check the data set'))
        self.img_loader = default_loader
        self.mask_loader = mask_loader
        self.transform = transform

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        img = self.img_loader(img_path)
        img = img.resize((288, 288))
        mask = self.mask_loader(mask_path)
        mask = mask.resize((288, 288), Image.NEAREST)
        if self.transform is not None:
            img = self.transform(img)
        mask = torch.from_numpy(np.array(mask, dtype=np.int32)).long()

        return img, mask

    def __len__(self):
        return len(self.imgs)
