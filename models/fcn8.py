import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from utils import initialize_weights
from .config import vgg19_bn_path, res152_path, dense201_path


class _FCN8Base(nn.Module):
    def __init__(self):
        super(_FCN8Base, self).__init__()
        self.features3 = None
        self.features4 = None
        self.features5 = None
        self.score3 = None
        self.score4 = None
        self.score5 = None

    def forward(self, x):
        y3 = self.features3(x)
        y4 = self.features4(y3)
        y5 = self.features5(y4)

        y5 = self.score5(y5)
        y4 = self.score4(y4)
        y3 = self.score3(y3)

        y = y4 + F.upsample(y5, y4.size()[2:], mode='bilinear')
        y = y3 + F.upsample(y, y3.size()[2:], mode='bilinear')
        y = F.upsample(y, x.size()[2:], mode='bilinear')
        return y


class FCN8VGG(_FCN8Base):
    def __init__(self, num_classes, pretrained=True):
        super(FCN8VGG, self).__init__()
        vgg = models.vgg19_bn()
        if pretrained:
            vgg.load_state_dict(torch.load(vgg19_bn_path))
        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())
        features[0].padding = (100, 100)
        for f in features:
            if 'MaxPool' in f.__class__.__name__:
                f.ceil_mode = True
        self.features3 = nn.Sequential(*features[0:27])
        self.features4 = nn.Sequential(*features[27:40])
        self.features5 = nn.Sequential(*features[40:])
        self.score3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.score4 = nn.Conv2d(512, num_classes, kernel_size=1)
        conv_fc6 = nn.Conv2d(512, 4096, kernel_size=7)
        conv_fc6.weight.data.copy_(classifier[0].weight.data.view(4096, 512, 7, 7))
        conv_fc6.bias.data.copy_(classifier[0].bias.data)
        conv_fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        conv_fc7.weight.data.copy_(classifier[3].weight.data.view(4096, 4096, 1, 1))
        conv_fc7.bias.data.copy_(classifier[3].bias.data)
        score5 = nn.Conv2d(4096, num_classes, kernel_size=1)
        initialize_weights(self.score3, self.score4, score5)
        self.score5 = nn.Sequential(
            conv_fc6, nn.ReLU(), nn.Dropout(), conv_fc7, nn.ReLU(), nn.Dropout(), score5
        )


class FCN8ResNet(_FCN8Base):
    def __init__(self, num_classes, pretrained=True):
        super(FCN8ResNet, self).__init__()
        res = models.resnet152()
        if pretrained:
            res.load_state_dict(torch.load(res152_path))
        self.features3 = nn.Sequential(
            res.conv1, res.bn1, res.relu, res.maxpool, res.layer1, res.layer2
        )
        self.features4 = res.layer3
        self.features5 = res.layer4
        self.score3 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score4 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.score5 = nn.Conv2d(2048, num_classes, kernel_size=1)
        initialize_weights(self.score3, self.score4, self.score5)


class FCN8DenseNet(_FCN8Base):
    def __init__(self, num_classes, pretrained=True):
        super(FCN8DenseNet, self).__init__()
        dense = models.densenet201()
        if pretrained:
            dense.load_state_dict(torch.load(dense201_path))
        features = list(dense.features.children())
        self.features3 = nn.Sequential(*features[:8])
        self.features4 = nn.Sequential(*features[8:10])
        self.features5 = nn.Sequential(*features[10:])
        self.score3 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        self.score4 = nn.Sequential(
            nn.BatchNorm2d(896),
            nn.ReLU(inplace=True),
            nn.Conv2d(896, num_classes, kernel_size=1)
        )
        self.score5 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(1920, num_classes, kernel_size=1)
        )
        initialize_weights(self.score3, self.score4, self.score5)
