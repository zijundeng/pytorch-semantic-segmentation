# PyTorch for Semantic Segmentation
This repository contains some models for semantic segmentation and the pipeline of training and testing models, 
implemented in PyTorch.

## Models
1. Vanilla FCN: FCN32, FCN16, FCN8, in the versions of VGG, ResNet and DenseNet respectively.
2. U-Net
3. SegNet
4. PSPNet
5. GCN (global convolutional network)

## Visualization
Use powerful visualization of TensorBoard for PyTorch. [Here](https://github.com/lanpa/tensorboard-pytorch)  to install.

## Usage
1. Go to *models* directory and set the root path.
2. Go to *datasets* directory and do following the README.
3. Adjust the argument settings in *train_psp.py* (or train_fcn8.py, train_gcn.py) and run it.

## Reference
I have borrowed some code from these nice repositories: [[1]](https://github.com/bodokaiser/piwise),
[[2]](https://github.com/ycszen/pytorch-ss). Thank them for the sharing.

## TODO
1. DeepLab v3
2. RefineNet
3. CRFAsRNN
4. Some evaluation criterion (e.g. mIOU)
5. More dataset
