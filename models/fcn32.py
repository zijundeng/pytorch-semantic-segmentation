import torch
from torch import nn
from torchvision import models

from utils import initialize_weights, get_upsampling_weight
from .config import vgg16_path, res152_path, dense201_path


class _FCN32Base(nn.Module):
    def __init__(self, num_classes):
        super(_FCN32Base, self).__init__()
        self.features5 = None
        self.score5 = None
        self.upscore5 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32, bias=False)
        self.upscore5.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 64))

    def forward(self, x):
        y = self.features5(x)
        y = self.score5(y)
        y_up = self.upscore5(y)
        return y_up[:, :, 19: (19 + x.size()[2]), 19: (19 + x.size()[3])].contiguous()


class FCN32VGG(_FCN32Base):
    def __init__(self, num_classes, pretrained=True):
        super(FCN32VGG, self).__init__(num_classes)
        vgg = models.vgg16()
        if pretrained:
            vgg.load_state_dict(torch.load(vgg16_path))
        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())
        features[0].padding = (100, 100)
        for f in features:
            if 'MaxPool' in f.__class__.__name__:
                f.ceil_mode = True
            elif 'ReLU' in f.__class__.__name__:
                f.inplace = True
        self.features5 = nn.Sequential(*features)

        conv_fc6 = nn.Conv2d(512, 4096, kernel_size=7)
        conv_fc6.weight.data.copy_(classifier[0].weight.data.view(4096, 512, 7, 7))
        conv_fc6.bias.data.copy_(classifier[0].bias.data)
        conv_fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        conv_fc7.weight.data.copy_(classifier[3].weight.data.view(4096, 4096, 1, 1))
        conv_fc7.bias.data.copy_(classifier[3].bias.data)
        score5 = nn.Conv2d(4096, num_classes, kernel_size=1)
        score5.weight.data.zero_()
        score5.bias.data.zero_()
        self.score5 = nn.Sequential(
            conv_fc6, nn.ReLU(inplace=True), nn.Dropout(),
            conv_fc7, nn.ReLU(inplace=True), nn.Dropout(),
            score5
        )


class FCN32ResNet(_FCN32Base):
    def __init__(self, num_classes, pretrained=True):
        super(FCN32ResNet, self).__init__(num_classes)
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
        super(FCN32DenseNet, self).__init__(num_classes)
        dense = models.densenet201()
        if pretrained:
            dense.load_state_dict(torch.load(dense201_path))
        self.features5 = dense.features
        self.score5 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(1920, num_classes, kernel_size=1)
        )
        initialize_weights(self.score5)
