import os

from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

from configuration import num_classes, ckpt_path, predict_path
from datasets import CityScapes
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
    simultaneous_transform = SimultaneousCompose([
        SimultaneousScale(585),
        SimultaneousRandomCrop((512, 1024)),
        SimultaneousRandomHorizontallyFlip()
    ])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*mean_std)
    ])
    restore = transforms.Compose([
        DeNormalize(*mean_std),
        transforms.ToPILImage()
    ])

    dataset = CityScapes('val', simultaneous_transform=simultaneous_transform, transform=transform,
                         target_transform=MaskToTensor())
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=16, shuffle=True)

    if not os.path.exists(predict_path):
        os.mkdir(predict_path)

    batch_inputs = []
    batch_outputs = []
    batch_labels = []
    for vi, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs = Variable(inputs, volatile=True).cuda()
        labels = Variable(labels, volatile=True).cuda()

        outputs = net(inputs)

        batch_inputs.append(inputs.cpu())
        batch_outputs.append(outputs.cpu())
        batch_labels.append(labels.cpu())

    batch_inputs = torch.cat(batch_inputs)
    batch_outputs = torch.cat(batch_outputs)
    batch_labels = torch.cat(batch_labels)

    batch_inputs = batch_inputs.data
    batch_outputs = batch_outputs.data
    batch_labels = batch_labels.data.numpy()
    batch_prediction = batch_outputs.max(1)[1].squeeze_(1).numpy()

    for idx, tensor in enumerate(zip(batch_inputs, batch_prediction, batch_labels)):
        pil_input = restore(tensor[0])
        pil_output = colorize_cityscapes_mask(tensor[1])
        pil_label = colorize_cityscapes_mask(tensor[2])
        pil_input.save(os.path.join(predict_path, '%d_img.png' % idx))
        pil_output.save(os.path.join(predict_path, '%d_out.png' % idx))
        pil_label.save(os.path.join(predict_path, '%d_label.png' % idx))


if __name__ == '__main__':
    main()
