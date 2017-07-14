import os
from PIL import Image

from torch.utils import data

from configuration import fine_train_path, fine_val_path


def default_loader(path):
    return Image.open(path)


def make_dataset(mode):
    images = []
    if mode == 'train':
        fine_train_img_path = os.path.join(fine_train_path, 'img')
        fine_train_mask_path = os.path.join(fine_train_path, 'mask')
        for img_name in [img_name.split('leftImg8bit.png')[0] for img_name in os.listdir(fine_train_img_path)]:
            item = (os.path.join(fine_train_img_path, img_name + 'leftImg8bit.png'),
                    os.path.join(fine_train_mask_path, img_name + 'gtFine_labelIds.png'))
            images.append(item)
    elif mode == 'val':
        fine_val_img_path = os.path.join(fine_val_path, 'img')
        fine_val_mask_path = os.path.join(fine_val_path, 'mask')
        for img_name in [img_name.split('leftImg8bit.png')[0] for img_name in os.listdir(fine_val_img_path)]:
            item = (os.path.join(fine_val_img_path, img_name + 'leftImg8bit.png'),
                    os.path.join(fine_val_mask_path, img_name + 'gtFine_labelIds.png'))
            images.append(item)
    return images


class CityScapes(data.Dataset):
    def __init__(self, mode, simultaneous_transform=None, transform=None, target_transform=None):
        self.imgs = make_dataset(mode)
        if len(self.imgs) == 0:
            raise (RuntimeError('Found 0 images, please check the data set'))
        self.loader = default_loader
        self.simultaneous_transform = simultaneous_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        img = self.loader(img_path)
        mask = self.loader(mask_path)
        if self.simultaneous_transform is not None:
            img, mask = self.simultaneous_transform(img, mask)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return img, mask

    def __len__(self):
        return len(self.imgs)
