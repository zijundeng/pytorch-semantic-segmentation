import math

import numpy as np
from torch import nn


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2. / n))
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                n = module.weight.size(1)
                module.weight.data.normal_(0, math.sqrt(2. / n))
                module.bias.data.zero_()


def adjust_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print 'lr is set to', lr


# color map
label_colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
                (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
                (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]
# 0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle # 6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=diningtable,
# 12=dog, 13=horse, 14=motorbike, 15=person # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor


def colorize_mask(mask, ignored_label):
    mask = mask.astype(np.uint8)
    h, w = mask.shape
    cmap = np.zeros((h, w, 3))

    for i in xrange(h):
        for j in xrange(w):
            v = mask[i, j]
            if v == ignored_label:
                continue
            cmap[i, j, :] = label_colors[v]

    return cmap.astype(np.uint8)


def calculate_mean_iu(predictions, gts, num_classes, ignored_label):
    sum_iu = 0
    for i in xrange(num_classes):
        n_ii = t_i = sum_n_ji = 0
        for p, gt in zip(predictions, gts):
            n_ii += np.sum(gt[p == i] == i)
            t_i += np.sum(gt == i)
            sum_n_ji += np.sum(p == i)
        sum_iu += float(n_ii) / (t_i + sum_n_ji - n_ii)
    mean_iu = sum_iu / num_classes
    return mean_iu

