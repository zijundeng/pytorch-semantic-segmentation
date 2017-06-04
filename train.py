import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from configuration import *
from datasets import VOC
from models.vanilla_fcn import VGG
from utils import *

tensor_to_pil = transforms.ToPILImage()


def main():
    training_batch_size = 16
    epoch_num = 100
    iter_freq_print_training_log = 500
    lr = 1e-2

    net = VGG(pretrained=True, num_classes=num_classes).cuda()
    net.train()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    de_normalize = DeNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    train_set = VOC(voc_dataset_root, transform)
    train_loader = DataLoader(train_set, batch_size=training_batch_size, num_workers=8, shuffle=True)

    criterion = CrossEntropyLoss2d().cuda()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, nesterov=True)

    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)

    for epoch in range(0, epoch_num):
        train(train_loader, net, criterion, optimizer, epoch, iter_freq_print_training_log, de_normalize)
        if (epoch + 1) % 8 == 0:
            lr /= 2
            optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, nesterov=True)


def train(train_loader, net, criterion, optimizer, epoch, iter_freq_print_training_log, de_normalize):
    epoch_dir = os.path.join(ckpt_path, str(epoch + 1))
    if not os.path.exists(epoch_dir):
        os.mkdir(epoch_dir)
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % iter_freq_print_training_log == 0:
            print '[epoch %d], [iter %d], [training_batch_loss %.4f]' % (epoch + 1, i + 1, loss.data[0])
            inputs = inputs.cpu().data
            outputs = outputs.cpu().data
            loss = loss.data[0]
            torch.save(net.state_dict(), ckpt_path + '/epoch_%d_iter_%d_loss_%.4f.pth' % (epoch + 1, i + 1, loss))
            to_save_dir = os.path.join(ckpt_path, str(epoch + 1), str(i + 1))
            os.mkdir(to_save_dir)
            for idx, tensor in enumerate(zip(inputs, outputs.max(1)[1].squeeze_(1))):
                pil_input = tensor_to_pil(de_normalize(tensor[0]))
                # tensor[1][tensor[1] > 0] = 255
                pil_output = Image.fromarray(tensor[1].numpy().astype('uint8'), 'P')
                pil_input.save(os.path.join(to_save_dir, '%d_img.png' % idx))
                pil_output.save(os.path.join(to_save_dir, '%d_out.png' % idx))


if __name__ == '__main__':
    main()
