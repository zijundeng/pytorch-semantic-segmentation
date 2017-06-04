import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from configuration import pretrained_vgg19, pretrained_res152, pretrained_dense201
from utils.training import initialize_weights


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
        vgg = models.vgg19()
        if pretrained:
            vgg.load_state_dict(torch.load(pretrained_vgg19))
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
            res.load_state_dict(torch.load(pretrained_res152))
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
            dense.load_state_dict(torch.load(pretrained_dense201))
        self.features5 = dense.features
        self.fconv5 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(1920, num_classes, kernel_size=7)
        )
        initialize_weights(self.fconv5)


        # from torch.autograd import Variable
        # import time
        # net = FCN32DenseNet(pretrained=True, num_classes=21).cuda()
        # inputs = Variable(torch.randn((8, 3, 512, 320))).cuda()
        # a = time.time()
        # outputs = net(inputs)
        # print time.time() - a
        # print outputs.size()
        # print models.densenet201()
