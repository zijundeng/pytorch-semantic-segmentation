import numpy as np

from utils.io import rmrf_mkdir
from .config import *

val_percentage = 0.02

img_list = [os.path.splitext(img)[0] for img in os.listdir(raw_mask_path)]
img_list = np.random.permutation(img_list)

val_data_num = int(len(img_list) * val_percentage)
val_data = img_list[:val_data_num]
train_data = img_list[val_data_num:]

rmrf_mkdir(processed_train_path)
rmrf_mkdir(processed_val_path)

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
