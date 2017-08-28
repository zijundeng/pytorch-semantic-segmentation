import torch
from torch import nn
from torchvision import models

from utils import initialize_weights, get_upsampling_weight
from .config import vgg16_path, res152_path, dense201_path


class _FCN8Base(nn.Module):
    def __init__(self, num_classes):
        super(_FCN8Base, self).__init__()
        self.features3 = None
        self.features4 = None
        self.features5 = None
        self.score3 = None
        self.score4 = None
        self.score5 = None
        self.upscore5 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore5_4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore5_4_3 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, bias=False)
        self.upscore5.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 4))
        self.upscore5_4.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 4))
        self.upscore5_4_3.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 16))

    def forward(self, x):
        y3 = self.features3(x)
        y4 = self.features4(y3)
        y5 = self.features5(y4)

        y5 = self.score5(y5)
        y5_up = self.upscore5(y5)

        y4 = self.score4(0.01 * y4)
        y5_4_up = self.upscore5_4(y4[:, :, 5: (5 + y5_up.size()[2]), 5: (5 + y5_up.size()[3])] + y5_up)

        y3 = self.score3(0.0001 * y3)
        y5_4_3_up = self.upscore5_4_3(y3[:, :, 9: (9 + y5_4_up.size()[2]), 9: (9 + y5_4_up.size()[3])] + y5_4_up)
        return y5_4_3_up[:, :, 31: (31 + x.size()[2]), 31: (31 + x.size()[3])].contiguous()


class FCN8VGG(_FCN8Base):
    def __init__(self, num_classes, pretrained=True):
        super(FCN8VGG, self).__init__(num_classes)
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
        self.features3 = nn.Sequential(*features[: 17])
        self.features4 = nn.Sequential(*features[17: 24])
        self.features5 = nn.Sequential(*features[24:])

        self.score3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.score4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score3.weight.data.zero_()
        self.score3.bias.data.zero_()
        self.score4.weight.data.zero_()
        self.score4.bias.data.zero_()

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
            conv_fc6, nn.ReLU(inplace=True), nn.Dropout(inplace=True),
            conv_fc7, nn.ReLU(inplace=True), nn.Dropout(inplace=True),
            score5
        )


class FCN8ResNet(_FCN8Base):
    def __init__(self, num_classes, pretrained=True):
        super(FCN8ResNet, self).__init__(num_classes)
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
        super(FCN8DenseNet, self).__init__(num_classes)
        dense = models.densenet201()
        if pretrained:
            dense.load_state_dict(torch.load(dense201_path))
        features = list(dense.features.children())
        self.features3 = nn.Sequential(*features[: 8])
        self.features4 = nn.Sequential(*features[8: 10])
        self.features5 = nn.Sequential(*features[10:])
        self.score3 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        self.score4 = nn.Sequential(
            nn.BatchNorm2d(896),
            nn.ReLU(),
            nn.Conv2d(896, num_classes, kernel_size=1)
        )
        self.score5 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(1920, num_classes, kernel_size=1)
        )
        initialize_weights(self.score3, self.score4, self.score5)
