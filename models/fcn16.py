import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from configuration import pretrained_vgg19, pretrained_res152, pretrained_dense201
from utils.training import initialize_weights


class _FCN16Base(nn.Module):
    def __init__(self):
        super(_FCN16Base, self).__init__()
        self.features4 = None
        self.features5 = None
        self.fconv4 = None
        self.fconv5 = None

    def forward(self, x):
        y4 = self.features4(x)
        y5 = self.features5(y4)
        y5 = self.fconv5(y5)
        y4 = self.fconv4(y4)
        y = y4 + F.upsample_bilinear(y5, y4.size()[2:])
        y = F.upsample_bilinear(y, x.size()[2:])
        return y


class FCN16VGG(_FCN16Base):
    def __init__(self, pretrained, num_classes):
        super(FCN16VGG, self).__init__()
        vgg = models.vgg19()
        if pretrained:
            vgg.load_state_dict(torch.load(pretrained_vgg19))
        features = list(vgg.features.children())
        self.features4 = nn.Sequential(*features[0:28])
        self.features5 = nn.Sequential(*features[28:])
        self.fconv4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.fconv5 = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, num_classes, kernel_size=1)
        )
        initialize_weights(self.fconv4, self.fconv5)


class FCN16ResNet(_FCN16Base):
    def __init__(self, pretrained, num_classes):
        super(FCN16ResNet, self).__init__()
        res = models.resnet152()
        if pretrained:
            res.load_state_dict(torch.load(pretrained_res152))
        self.features4 = nn.Sequential(
            res.conv1, res.bn1, res.relu, res.maxpool, res.layer1, res.layer2, res.layer3
        )
        self.features5 = res.layer4
        self.fconv4 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.fconv5 = nn.Conv2d(2048, num_classes, kernel_size=7)
        initialize_weights(self.fconv4, self.fconv5)


class FCN16DenseNet(_FCN16Base):
    def __init__(self, pretrained, num_classes):
        super(FCN16DenseNet, self).__init__()
        dense = models.densenet201()
        if pretrained:
            dense.load_state_dict(torch.load(pretrained_dense201))
        features = list(dense.features.children())
        self.features4 = nn.Sequential(*features[:10])
        self.features5 = nn.Sequential(*features[10:])
        self.fconv4 = nn.Sequential(
            nn.BatchNorm2d(896),
            nn.ReLU(inplace=True),
            nn.Conv2d(896, num_classes, kernel_size=1)
        )
        self.fconv5 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(1920, num_classes, kernel_size=7)
        )
        initialize_weights(self.fconv4, self.fconv5)
