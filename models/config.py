import os

# here (https://github.com/pytorch/vision/tree/master/torchvision/models) to find the download link of pretrained models

root = '/media/library/Packages/Models/PyTorch Pretrained'
res152_path = os.path.join(root, 'ResNet', 'resnet152-b121ed2d.pth')
inception_v3_path = os.path.join(root, 'Inception', 'inception_v3_google-1a9a5a14.pth')
vgg19_bn_path = os.path.join(root, 'VggNet', 'vgg19_bn-c79401a0.pth')
dense201_path = os.path.join(root, 'DenseNet', 'densenet201-4c113574.pth')
