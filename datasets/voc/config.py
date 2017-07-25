import os

root = '/home/b3-542/datasets/VOCdevkit/VOC2012'
# download SegmentationClassAug, place it to VOCdevkit/VOC2012/SegmentationClassAug
# download link: https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0
# Thank TheLegendAli for the download link
raw_img_path = os.path.join(root, 'JPEGImages')
raw_mask_path = os.path.join(root, 'SegmentationClassAug')

processed_train_path = os.path.join(root, 'train')
processed_val_path = os.path.join(root, 'val')
image_dir_name = 'images'
mask_dir_name = 'masks'

num_classes = 20
ignored_label = 255

'''
color map
0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle # 6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=diningtable,
12=dog, 13=horse, 14=motorbike, 15=person # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
'''
label_colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
                (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
                (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]
