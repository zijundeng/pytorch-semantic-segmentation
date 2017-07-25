import os

from PIL import Image
from torch.utils import data

from config import processed_train_path, processed_val_path, image_dir_name, mask_dir_name


def default_loader(path):
    return Image.open(path)


def make_dataset(mode):
    if mode == 'train':
        root = processed_train_path
    elif mode == 'val':
        root = processed_val_path
    else:
        raise ValueError('the mode should be either train or val!')
    mask_dir_path = os.path.join(root, mask_dir_name)
    image_dir_path = os.path.join(root, image_dir_name)
    images = []
    for img_name in [os.path.splitext(img_name)[0] for img_name in os.listdir(mask_dir_path)]:
        item = (os.path.join(image_dir_path, img_name + '.jpg'), os.path.join(mask_dir_path, img_name + '.png'))
        images.append(item)
    return images


class VOC(data.Dataset):
    def __init__(self, mode, simul_transform=None, transform=None, target_transform=None):
        self.imgs = make_dataset(mode)
        if len(self.imgs) == 0:
            raise (RuntimeError('Found 0 images, please check the data set'))
        self.loader = default_loader
        self.simul_transform = simul_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        img = self.loader(img_path)
        mask = self.loader(mask_path)
        if self.simul_transform is not None:
            img, mask = self.simul_transform(img, mask)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return img, mask

    def __len__(self):
        return len(self.imgs)
