import os

# official pretrained models
pretrained_root = '/media/b3-542/LIBRARY/ZijunDeng/PyTorch Pretrained'
pretrained_res152 = os.path.join(pretrained_root, 'ResNet', 'resnet152-b121ed2d.pth')
pretrained_inception_v3 = os.path.join(pretrained_root, 'Inception', 'inception_v3_google-1a9a5a14.pth')
pretrained_vgg19 = os.path.join(pretrained_root, 'VggNet', 'vgg19-dcbb9e9d.pth')
pretrained_dense201 = os.path.join(pretrained_root, 'DenseNet', 'densenet201-4c113574.pth')

voc_dataset_root = '/home/b3-542/datasets/VOCdevkit/VOC2012'
# download SegmentationClassAug, place it to VOCdevkit/VOC2012/SegmentationClassAug
# download link: https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0
# Thank TheLegendAli for the download link
voc_image_dir_path = os.path.join(voc_dataset_root, 'JPEGImages')
voc_mask_dir_path = os.path.join(voc_dataset_root, 'SegmentationClassAug')

train_path = os.path.join(voc_dataset_root, 'segmentation-train')
val_path = os.path.join(voc_dataset_root, 'segmentation-val')
image_dir_name = 'images'
mask_dir_name = 'masks'

num_classes = 21

ckpt_path = './ckpt'
