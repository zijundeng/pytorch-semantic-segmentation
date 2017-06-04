import math

import torch
from torch import nn
from torchvision import models

from configuration import *


def _initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            n = m.weight.size(1)
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()


class VGG(nn.Module):
    def __init__(self, pretrained, num_classes):
        super(VGG, self).__init__()
        vgg = models.vgg19()
        if pretrained:
            vgg.load_state_dict(torch.load(pretrained_vgg19))
        self.features = vgg.features
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Conv2d(4096, num_classes, kernel_size=1),
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=96, stride=96)
        )
        _initialize_weights(self.classifier)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class Inception(nn.Module):
    # it is hard to keep the output size of inception unchanged and hence inception is not convenient
    def __init__(self, pretrained, num_classes):
        super(Inception, self).__init__()
        inception = models.inception_v3()
        if pretrained:
            inception.load_state_dict(torch.load(pretrained_inception_v3))
        self.features = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3, inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2), inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2), inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d,
            inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c, inception.Mixed_6d, inception.Mixed_6e,
            inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(2048, num_classes, kernel_size=1),
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=32, stride=32)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResNet(nn.Module):
    def __init__(self, pretrained, num_classes):
        super(ResNet, self).__init__()
        res = models.resnet152()
        if pretrained:
            res.load_state_dict(torch.load(pretrained_res152))
        self.features = nn.Sequential(
            res.conv1, res.bn1, res.relu, res.maxpool, res.layer1, res.layer2, res.layer3, res.layer4
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(2048, num_classes, kernel_size=1),
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=32, stride=32)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class DenseNet(nn.Module):
    def __init__(self, pretrained, num_classes):
        super(DenseNet, self).__init__()
        dense = models.densenet201()
        if pretrained:
            dense.load_state_dict(torch.load(pretrained_dense201))
        self.features = nn.Sequential(
            dense.features,
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(1920, num_classes, kernel_size=1),
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=32, stride=32)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
