import torch.nn.functional as F
from torch import nn


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)


class CrossEntropyLoss2dOld(nn.Module):
    def __init__(self, ignored_label, size_average=True):
        super(CrossEntropyLoss2dOld, self).__init__()
        self.ignored_label = ignored_label
        self.size_average = size_average

    def forward(self, inputs, targets):
        n, c, h, w = inputs.size()
        inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous()

        if self.ignored_label is None:
            inputs = inputs.view(-1, c)
            targets = targets.view(-1)
        else:
            useful_idx = targets != self.ignored_label
            inputs = inputs[useful_idx.repeat(1, 1, 1, c)].view(-1, c)
            targets = targets[useful_idx].view(-1)

        return F.cross_entropy(inputs, targets, self.weight, self.size_average)
