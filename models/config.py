import os

# PyTorch will automatically download pretrained weights into `os.environ['TORCH_MODEL_ZOO']`
# using the mechanism described here: (http://pytorch.org/docs/master/model_zoo.html)
# Download links used are also listed here: (https://github.com/pytorch/vision/tree/master/torchvision/models)

'''
vgg16 trained using caffe
visit this (https://github.com/jcjohnson/pytorch-vgg) to download the converted vgg16
'''
vgg16_caffe_path = os.path.join(os.environ.get('TORCH_MODEL_ZOO', '.'), 'vgg16-caffe.pth')
