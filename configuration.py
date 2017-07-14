import os

# ----------------------------------------------------------------------------------------------------------------------
"""
The following paths in this block should be correctly referred
"""
# go here (https://github.com/pytorch/vision/tree/master/torchvision/models) to find the download link
# if you just use vgg-fcn, just only download the vgg pretrained model
pretrained_root = '/media/b3-542/LIBRARY/ZijunDeng/PyTorch Pretrained'  # should be correctly referred
pretrained_res152 = os.path.join(pretrained_root, 'ResNet', 'resnet152-b121ed2d.pth')
pretrained_inception_v3 = os.path.join(pretrained_root, 'Inception', 'inception_v3_google-1a9a5a14.pth')
pretrained_vgg19_bn = os.path.join(pretrained_root, 'VggNet', 'vgg19_bn-c79401a0.pth')
pretrained_dense201 = os.path.join(pretrained_root, 'DenseNet', 'densenet201-4c113574.pth')

voc_dataset_root = '/home/b3-542/datasets/VOCdevkit/VOC2012'  # should be correctly referred
# download SegmentationClassAug, place it to VOCdevkit/VOC2012/SegmentationClassAug
# download link: https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0
# Thank TheLegendAli for the download link
voc_image_dir_path = os.path.join(voc_dataset_root, 'JPEGImages')
voc_mask_dir_path = os.path.join(voc_dataset_root, 'SegmentationClassAug')
ckpt_path = '/media/b3-542/LIBRARY/ZijunDeng/ckpt'  # path to stored checkpoints

cityscapes_dataset_root = '/home/b3-542/cityscapes'
leftImg8bit_path = os.path.join(cityscapes_dataset_root, 'leftImg8bit_trainvaltest/leftImg8bit')
gtFine_trainvaltest_path = os.path.join(cityscapes_dataset_root, 'gtFine_trainvaltest/gtFine')
fine_path = os.path.join(cityscapes_dataset_root, 'preprocessed')
fine_train_path = os.path.join(fine_path, 'train')
fine_val_path = os.path.join(fine_path, 'val')
# ----------------------------------------------------------------------------------------------------------------------

"""
You don't need to modify the things below.
"""
train_path = os.path.join(voc_dataset_root, 'segmentation-train')
val_path = os.path.join(voc_dataset_root, 'segmentation-val')
image_dir_name = 'images'
mask_dir_name = 'masks'

num_classes = 19
ignored_label = 255
