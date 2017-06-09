import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_features, out_features, down_scale, up_size):
        super(PyramidPoolingModule, self).__init__()

        self.features = nn.Sequential(
            nn.AvgPool2d(down_scale, stride=down_scale),
            nn.Conv2d(in_features, out_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_features, momentum=.95),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(up_size)
        )

    def forward(self, x):
        return self.features(x)


class PSPNet(nn.Module):
    def __init__(self, pretrained, num_classes):
        super(PSPNet, self).__init__()

        resnet = models.resnet101(pretrained=True)

        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=.95),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.stride = 1
                m.requires_grad = False
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad = False

        self.layer5a = PyramidPoolingModule(2048, 512, 60, 60)
        self.layer5b = PyramidPoolingModule(2048, 512, 30, 60)
        self.layer5c = PyramidPoolingModule(2048, 512, 20, 60)
        self.layer5d = PyramidPoolingModule(2048, 512, 10, 60)

        self.final = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(.1),
            nn.Conv2d(512, num_classes, 1),
        )

    def forward(self, x):
        y = self.conv1(x)
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = self.final(torch.cat([
            y,
            self.layer5a(y),
            self.layer5b(y),
            self.layer5c(y),
            self.layer5d(y),
        ], 1))

        return F.upsample_bilinear(y, x.size()[2:])
