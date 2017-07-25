import shutil

import numpy as np
from PIL import Image

from datasets.cityscapes.config import *
from utils.io import rmrf_mkdir
from .config import ignored_label
from .utils import colorize_mask

'''
see https://github.com/mcordts/cityscapesScripts/blob/a81df1242c8f047d03219b9b5fe22436e4c02687/cityscapesscripts/
helpers/labels.py
'''
id_to_trainid = {-1: ignored_label, 0: ignored_label, 1: ignored_label, 2: ignored_label, 3: ignored_label,
                 4: ignored_label, 5: ignored_label, 6: ignored_label, 7: 0, 8: 1, 9: ignored_label,
                 10: ignored_label, 11: 2, 12: 3, 13: 4, 14: ignored_label, 15: ignored_label, 16: ignored_label,
                 17: 5, 18: ignored_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                 28: 15, 29: ignored_label, 30: ignored_label, 31: 16, 32: 17, 33: 18}

raw_train_img = os.path.join(raw_img_path, 'train')
raw_train_mask = os.path.join(raw_mask_path, 'train')
raw_val_img = os.path.join(raw_img_path, 'val')
raw_val_mask = os.path.join(raw_mask_path, 'val')

processed_train_img = os.path.join(processed_train_path, 'img')
processed_train_mask = os.path.join(processed_train_path, 'mask')
processed_val_img = os.path.join(processed_val_path, 'img')
processed_val_mask = os.path.join(processed_val_path, 'mask')

rmrf_mkdir(processed_path)
rmrf_mkdir(processed_train_path)
rmrf_mkdir(processed_train_img)
rmrf_mkdir(processed_train_mask)
rmrf_mkdir(processed_val_path)
rmrf_mkdir(processed_val_img)
rmrf_mkdir(processed_val_mask)

for d in os.listdir(raw_train_img):
    cate_img_dir = os.path.join(raw_train_img, d)
    cate_mask_dir = os.path.join(raw_train_mask, d)
    for img_name in os.listdir(cate_img_dir):
        shutil.copy(os.path.join(cate_img_dir, img_name), processed_train_img)
    for mask_name in os.listdir(cate_mask_dir):
        if not mask_name.endswith('labelIds.png'):
            continue
        mask = Image.open(os.path.join(cate_mask_dir, mask_name))
        mask = np.array(mask)
        new_mask = mask.copy()
        for k, v in id_to_trainid.iteritems():
            new_mask[mask == k] = v
        new_mask = colorize_mask(new_mask)
        new_mask.save(os.path.join(processed_train_mask, mask_name))

for d in os.listdir(raw_val_img):
    cate_img_dir = os.path.join(raw_val_img, d)
    cate_mask_dir = os.path.join(raw_val_mask, d)
    for img_name in os.listdir(cate_img_dir):
        shutil.copy(os.path.join(cate_img_dir, img_name), processed_val_img)
    for mask_name in os.listdir(cate_mask_dir):
        if not mask_name.endswith('labelIds.png'):
            continue
        mask = Image.open(os.path.join(cate_mask_dir, mask_name))
        mask = np.array(mask)
        new_mask = mask.copy()
        for k, v in id_to_trainid.iteritems():
            new_mask[mask == k] = v
        new_mask = colorize_mask(new_mask)
        new_mask.save(os.path.join(processed_val_mask, mask_name))
