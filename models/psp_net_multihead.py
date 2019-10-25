import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from ..utils import initialize_weights

from .psp_net import _PyramidPoolingModule


class PSPHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.ppm = _PyramidPoolingModule(2048, 512, (1, 2, 3, 6))
        self.final = nn.Sequential(
            nn.Conv2d(4096, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

        initialize_weights(self.ppm, self.final)

    def forward(self, features_from_backbone, img_size):
        result = self.final(self.ppm(features_from_backbone))
        return F.interpolate(result, img_size[2:], mode='bilinear')


class PSPNet_Multihead(nn.Module):
    def __init__(self, num_heads, num_classes, pretrained=True):
        super().__init__()

        self.init_heads(num_heads, num_classes=num_classes)
        self.init_backbone(pretrained=pretrained)

    def init_backbone(self, pretrained):
        resnet = models.resnet101(pretrained=pretrained)

        for n, m in resnet.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in resnet.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        self.backbone = nn.Sequential(
            nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool), # layer 0
            resnet.layer1, 
            resnet.layer2,
            resnet.layer3, 
            resnet.layer4,
        )

    def init_heads(self, num_heads, num_classes):

        self.heads = nn.Sequential(
            *[PSPHead(num_classes=num_classes) for i in range(num_heads)]
        )

    def forward(self, image):
        img_size = image.size()

        backbone_features = self.backbone(image)

        return torch.cat([
            head(backbone_features, img_size=img_size)
            for head in self.heads
        ], dim=1)

