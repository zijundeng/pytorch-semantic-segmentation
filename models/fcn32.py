import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from utils.training import initialize_weights
from .config import vgg19_bn_path, res152_path, dense201_path


class _FCN32Base(nn.Module):
    def __init__(self):
        super(_FCN32Base, self).__init__()
        self.features5 = None
        self.fconv5 = None

    def forward(self, x):
        y = self.features5(x)
        y = self.fconv5(y)
        y = F.upsample_bilinear(y, x.size()[2:])
        return y


class FCN32VGG(_FCN32Base):
    def __init__(self, pretrained, num_classes):
        super(FCN32VGG, self).__init__()
        vgg = models.vgg19_bn()
        if pretrained:
            vgg.load_state_dict(torch.load(vgg19_bn_path))
        self.features5 = vgg.features
        self.fconv5 = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, num_classes, kernel_size=1)
        )
        initialize_weights(self.fconv5)


class FCN32ResNet(_FCN32Base):
    def __init__(self, pretrained, num_classes):
        super(FCN32ResNet, self).__init__()
        res = models.resnet152()
        if pretrained:
            res.load_state_dict(torch.load(res152_path))
        self.features5 = nn.Sequential(
            res.conv1, res.bn1, res.relu, res.maxpool, res.layer1, res.layer2, res.layer3, res.layer4
        )
        self.fconv5 = nn.Conv2d(2048, num_classes, kernel_size=7)
        initialize_weights(self.fconv5)


class FCN32DenseNet(_FCN32Base):
    def __init__(self, pretrained, num_classes):
        super(FCN32DenseNet, self).__init__()
        dense = models.densenet201()
        if pretrained:
            dense.load_state_dict(torch.load(dense201_path))
        self.features5 = dense.features
        self.fconv5 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(1920, num_classes, kernel_size=7)
        )
        initialize_weights(self.fconv5)
