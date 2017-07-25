import os

root = '/home/b3-542/cityscapes'
raw_img_path = os.path.join(root, 'leftImg8bit_trainvaltest/leftImg8bit')
raw_mask_path = os.path.join(root, 'gtFine_trainvaltest/gtFine')

processed_path = os.path.join(root, 'processed')
processed_train_path = os.path.join(processed_path, 'train')
processed_val_path = os.path.join(processed_path, 'val')

num_classes = 19
ignored_label = 255

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)
