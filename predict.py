import os

from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import LSUN

from configuration import num_classes, ckpt_path, predict_path
from models import PSPNet
from utils.training import colorize_cityscapes_mask
from utils.transforms import *

cudnn.benchmark = True


def main():
    batch_size = 8

    net = PSPNet(pretrained=False, num_classes=num_classes, input_size=(512, 1024)).cuda()
    snapshot = 'epoch_48_validation_loss_5.1326_mean_iu_0.3172_lr_0.00001000.pth'
    net.load_state_dict(torch.load(os.path.join(ckpt_path, snapshot)))
    net.eval()

    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Scale(512),
        transforms.ToTensor(),
        transforms.Normalize(*mean_std)
    ])
    restore = transforms.Compose([
        DeNormalize(*mean_std),
        transforms.ToPILImage()
    ])

    lsun_path = '/media/library/Packages/Datasets/LSUN'

    dataset = LSUN(lsun_path, 'test', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=16, shuffle=True)

    if not os.path.exists(predict_path):
        os.mkdir(predict_path)

    for vi, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs = Variable(inputs, volatile=True).cuda()
        labels = Variable(labels, volatile=True).cuda()
        outputs = net(inputs)

        prediction = outputs.cpu().data.max(1)[1].squeeze_(1).numpy()

        for idx, tensor in enumerate(zip(inputs.cpu().data, prediction, labels.cpu().data.numpy())):
            pil_input = restore(tensor[0])
            pil_output = colorize_cityscapes_mask(tensor[1])
            pil_label = colorize_cityscapes_mask(tensor[2])
            pil_input.save(os.path.join(predict_path, '%d_img.png' % idx))
            pil_output.save(os.path.join(predict_path, '%d_out.png' % idx))
            pil_label.save(os.path.join(predict_path, '%d_label.png' % idx))
            print 'save the #%d batch, %d images' % (vi + 1, idx + 1)


if __name__ == '__main__':
    main()
