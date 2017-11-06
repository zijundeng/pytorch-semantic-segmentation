import os

import torchvision.transforms as standard_transforms
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader

from datasets import voc
from models import *
from utils import check_mkdir

cudnn.benchmark = True

ckpt_path = './ckpt'

args = {
    'exp_name': 'voc-psp_net',
    'snapshot': 'epoch_33_loss_0.31766_acc_0.92188_acc-cls_0.81110_mean-iu_0.70271_fwavacc_0.86757_lr_0.0023769346.pth'
}


def main():
    net = PSPNet(num_classes=voc.num_classes).cuda()
    print('load model ' + args['snapshot'])
    net.load_state_dict(torch.load(os.path.join(ckpt_path, args['exp_name'], args['snapshot'])))
    net.eval()

    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    val_input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

    test_set = voc.VOC('test', transform=val_input_transform)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=8, shuffle=False)

    check_mkdir(os.path.join(ckpt_path, args['exp_name'], 'test'))

    for vi, data in enumerate(test_loader):
        img_name, img = data
        img_name = img_name[0]

        img = Variable(img, volatile=True).cuda()
        output = net(img)

        prediction = output.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
        prediction = voc.colorize_mask(prediction)
        prediction.save(os.path.join(ckpt_path, args['exp_name'], 'test', img_name + '.png'))

        print('%d / %d' % (vi + 1, len(test_loader)))


if __name__ == '__main__':
    main()
