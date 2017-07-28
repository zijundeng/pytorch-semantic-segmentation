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
1. Modify the **configuration.py** according to the hint in it.
2. Run **split_train_val.py**.
3. Set your model and training parameters in **train.py** and then run.

## Reference
I have borrowed some code from these nice repositories: [[1]](https://github.com/bodokaiser/piwise),
[[2]](https://github.com/ycszen/pytorch-ss). Thank them for the sharing.

## TODO
1. DeepLab v3
2. RefineNet
3. CRFAsRNN
4. Some evaluation criterion (e.g. mIOU)
5. More dataset
