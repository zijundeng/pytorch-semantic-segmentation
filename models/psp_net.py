import math

import torch
from torch import nn
from torchvision import models

from configuration import pretrained_res152


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_size, in_dim, reduction_dim, setting):
        super(PyramidPoolingModule, self).__init__()
        self.features = []
        for s in setting:
            pool_size = (math.ceil(float(in_size[0]) / s), math.ceil(float(in_size[1]) / s))
            self.features.append(nn.Sequential(
                nn.AvgPool2d(kernel_size=pool_size, stride=pool_size, ceil_mode=True),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim, momentum=.95),
                nn.ReLU(),
                nn.UpsamplingBilinear2d(size=in_size)
            ))

    def forward(self, x):
        out = [x]
        for f in self.features:
            out.append(f(x))
        out = torch.cat(out, 1)
        return out


class PSPNet(nn.Module):
    def __init__(self, pretrained, num_classes, input_size):
        super(PSPNet, self).__init__()
        resnet = models.resnet152()
        if pretrained:
            resnet.load_state_dict(torch.load(pretrained_res152))
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.ppm = PyramidPoolingModule((input_size[0] / 32, input_size[1] / 32), 2048, 512, (1, 2, 3, 6))

        self.final = nn.Sequential(
            nn.Conv2d(4096, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=.95),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.UpsamplingBilinear2d(size=input_size)
        )

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.ppm(x)
        x = self.final(x)
        return x


from torch.autograd import Variable
import time

net = PSPNet()
inputs = Variable(torch.randn((1, 3, 512, 1024)))
a = time.time()
outputs = net(inputs)
print time.time() - a
print outputs.size()
