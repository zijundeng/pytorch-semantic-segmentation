import os

import torch
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import LSUN

from datasets.cityscapes.config import num_classes
from datasets.cityscapes.utils import colorize_mask
from config import ckpt_path, predict_path
from models import PSPNet
import utils.transforms as expanded_transform

cudnn.benchmark = True


def main():
    batch_size = 8

    net = PSPNet(pretrained=False, num_classes=num_classes, input_size=(512, 1024)).cuda()
    snapshot = 'epoch_48_validation_loss_5.1326_mean_iu_0.3172_lr_0.00001000.pth'
    net.load_state_dict(torch.load(os.path.join(ckpt_path, snapshot)))
    net.eval()

    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform = transforms.Compose([
        expanded_transform.FreeScale((512, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(*mean_std)
    ])
    restore = transforms.Compose([
        expanded_transform.DeNormalize(*mean_std),
        transforms.ToPILImage()
    ])

    lsun_path = '/home/b3-542/LSUN'

    dataset = LSUN(lsun_path, ['tower_val', 'church_outdoor_val', 'bridge_val'], transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=16, shuffle=True)

    if not os.path.exists(predict_path):
        os.mkdir(predict_path)

    for vi, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs = Variable(inputs, volatile=True).cuda()
        outputs = net(inputs)

        prediction = outputs.cpu().data.max(1)[1].squeeze_(1).numpy()

        for idx, tensor in enumerate(zip(inputs.cpu().data, prediction)):
            pil_input = restore(tensor[0])
            pil_output = colorize_mask(tensor[1])
            pil_input.save(os.path.join(predict_path, '%d_img.png' % (vi * batch_size + idx)))
            pil_output.save(os.path.join(predict_path, '%d_out.png' % (vi * batch_size + idx)))
            print 'save the #%d batch, %d images' % (vi + 1, idx + 1)


if __name__ == '__main__':
    main()
