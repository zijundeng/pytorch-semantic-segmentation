import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss, _assert_no_grad


class CrossEntropyLoss2d(_WeightedLoss):
    def __init__(self, ignored_label, size_average):
        super(CrossEntropyLoss2d, self).__init__()
        self.ignored_label = ignored_label
        self.size_average = size_average

    def forward(self, inputs, targets):
        _assert_no_grad(targets)
        n, c, h, w = inputs.size()
        inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous()

        if self.ignored_label is None:
            inputs = inputs.view(-1, c)
            targets = targets.view(-1)
        else:
            # inputs = inputs[targets.view(n, h, w, 1).repeat(1, 1, 1, c) != self.ignored_label].view(-1, c)
            useful_idx = targets != self.ignored_label
            inputs = inputs[useful_idx.repeat(1, 1, 1, c)].view(-1, c)
            targets = targets[useful_idx].view(-1)

        return F.cross_entropy(inputs, targets, self.weight, self.size_average)
