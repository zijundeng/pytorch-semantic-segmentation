import os
import shutil

import numpy as np
from PIL import Image
from utils.training import colorize_cityscapes_mask

from configuration import leftImg8bit_path, gtFine_trainvaltest_path, fine_path, fine_train_path, fine_val_path
from utils.io import rmrf_mkdir

# see https://github.com/mcordts/cityscapesScripts/blob/a81df1242c8f047d03219b9b5fe22436e4c02687/cityscapesscripts/helpers/labels.py
id_to_trainid = {-1: 255, 0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0, 8: 1, 9: 255, 10: 255, 11: 2,
                 12: 3, 13: 4, 14: 255, 15: 255, 16: 255, 17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9,
                 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18}

ori_train_img = os.path.join(leftImg8bit_path, 'train')
ori_train_mask = os.path.join(gtFine_trainvaltest_path, 'train')
ori_val_img = os.path.join(leftImg8bit_path, 'val')
ori_val_mask = os.path.join(gtFine_trainvaltest_path, 'val')

fine_train_img = os.path.join(fine_train_path, 'img')
fine_train_mask = os.path.join(fine_train_path, 'mask')
fine_val_img = os.path.join(fine_val_path, 'img')
fine_val_mask = os.path.join(fine_val_path, 'mask')

rmrf_mkdir(fine_path)
rmrf_mkdir(fine_train_path)
rmrf_mkdir(fine_train_img)
rmrf_mkdir(fine_train_mask)
rmrf_mkdir(fine_val_path)
rmrf_mkdir(fine_val_img)
rmrf_mkdir(fine_val_mask)

for d in os.listdir(ori_train_img):
    cate_img_dir = os.path.join(ori_train_img, d)
    cate_mask_dir = os.path.join(ori_train_mask, d)
    for img_name in os.listdir(cate_img_dir):
        shutil.copy(os.path.join(cate_img_dir, img_name), fine_train_img)
    for mask_name in os.listdir(cate_mask_dir):
        if not mask_name.endswith('labelIds.png'):
            continue
        mask = Image.open(os.path.join(cate_mask_dir, mask_name))
        mask = np.array(mask)
        new_mask = mask.copy()
        for k, v in id_to_trainid.iteritems():
            new_mask[mask == k] = v
        new_mask = colorize_cityscapes_mask(new_mask)
        new_mask.save(os.path.join(fine_train_mask, mask_name))

for d in os.listdir(ori_val_img):
    cate_img_dir = os.path.join(ori_val_img, d)
    cate_mask_dir = os.path.join(ori_val_mask, d)
    for img_name in os.listdir(cate_img_dir):
        shutil.copy(os.path.join(cate_img_dir, img_name), fine_val_img)
    for mask_name in os.listdir(cate_mask_dir):
        if not mask_name.endswith('labelIds.png'):
            continue
        mask = Image.open(os.path.join(cate_mask_dir, mask_name))
        mask = np.array(mask)
        new_mask = mask.copy()
        for k, v in id_to_trainid.iteritems():
            new_mask[mask == k] = v
        new_mask = colorize_cityscapes_mask(new_mask)
        new_mask.save(os.path.join(fine_val_mask, mask_name))

