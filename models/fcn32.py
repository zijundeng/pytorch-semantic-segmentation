import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from utils import initialize_weights
from .config import vgg19_bn_path, res152_path, dense201_path


class _FCN32Base(nn.Module):
    def __init__(self):
        super(_FCN32Base, self).__init__()
        self.features5 = None
        self.score5 = None

    def forward(self, x):
        y = self.features5(x)
        y = self.score5(y)
        y = F.upsample(y, x.size()[2:], mode='bilinear')
        return y


class FCN32VGG(_FCN32Base):
    def __init__(self, num_classes, pretrained=True):
        super(FCN32VGG, self).__init__()
        vgg = models.vgg19_bn()
        if pretrained:
            vgg.load_state_dict(torch.load(vgg19_bn_path))
        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())
        features[0].padding = (100, 100)
        for f in features:
            if 'MaxPool' in f.__class__.__name__:
                f.ceil_mode = True
        self.features5 = nn.Sequential(*features)
        conv_fc6 = nn.Conv2d(512, 4096, kernel_size=7)
        conv_fc6.weight.data.copy_(classifier[0].weight.data.view(4096, 512, 7, 7))
        conv_fc6.bias.data.copy_(classifier[0].bias.data)
        conv_fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        conv_fc7.weight.data.copy_(classifier[3].weight.data.view(4096, 4096, 1, 1))
        conv_fc7.bias.data.copy_(classifier[3].bias.data)
        score5 = nn.Conv2d(4096, num_classes, kernel_size=1)
        initialize_weights(score5)
        self.score5 = nn.Sequential(
            conv_fc6, nn.ReLU(), nn.Dropout(), conv_fc7, nn.ReLU(), nn.Dropout(), score5
        )


class FCN32ResNet(_FCN32Base):
    def __init__(self, num_classes, pretrained=True):
        super(FCN32ResNet, self).__init__()
        res = models.resnet152()
        if pretrained:
            res.load_state_dict(torch.load(res152_path))
        self.features5 = nn.Sequential(
            res.conv1, res.bn1, res.relu, res.maxpool, res.layer1, res.layer2, res.layer3, res.layer4
        )
        self.score5 = nn.Conv2d(2048, num_classes, kernel_size=1)
        initialize_weights(self.score5)


class FCN32DenseNet(_FCN32Base):
    def __init__(self, num_classes, pretrained=True):
        super(FCN32DenseNet, self).__init__()
        dense = models.densenet201()
        if pretrained:
            dense.load_state_dict(torch.load(dense201_path))
        self.features5 = dense.features
        self.score5 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(1920, num_classes, kernel_size=1)
        )
        initialize_weights(self.score5)
