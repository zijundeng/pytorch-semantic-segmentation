import os
import shutil

import numpy as np

from configuration import voc_image_dir_path, voc_mask_dir_path, train_path, val_path, image_dir_name, mask_dir_name


def rmrf_mkdir(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.mkdir(dir_name)


val_percentage = 0.05

img_list = [os.path.splitext(img)[0] for img in os.listdir(voc_mask_dir_path)]
img_list = np.random.permutation(img_list)

val_data_num = int(len(img_list) * val_percentage)
val_data = img_list[:val_data_num]
train_data = img_list[val_data_num:]

rmrf_mkdir(train_path)
rmrf_mkdir(os.path.join(train_path, image_dir_name))
rmrf_mkdir(os.path.join(train_path, mask_dir_name))
rmrf_mkdir(val_path)
rmrf_mkdir(os.path.join(val_path, image_dir_name))
rmrf_mkdir(os.path.join(val_path, mask_dir_name))

for i, t in enumerate(train_data):
    os.symlink(os.path.join(voc_image_dir_path, t + '.jpg'),
               os.path.join(train_path, image_dir_name, t + '.jpg'))
    os.symlink(os.path.join(voc_mask_dir_path, t + '.png'),
               os.path.join(train_path, mask_dir_name, t + '.png'))
    print 'processed %d train samples' % i

for i, v in enumerate(val_data):
    os.symlink(os.path.join(voc_image_dir_path, v + '.jpg'),
               os.path.join(val_path, image_dir_name, v + '.jpg'))
    os.symlink(os.path.join(voc_mask_dir_path, v + '.png'),
               os.path.join(val_path, mask_dir_name, v + '.png'))
    print 'processed %d val samples' % i
