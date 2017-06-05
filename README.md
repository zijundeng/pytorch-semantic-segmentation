# PyTorch implementation of FCN-based models for semantic segmentation
This repository contains some FCN-based models and the pipeline of training and evaluating models.

## Models
1. Vanilla FCN: FCN32, FCN16, FCN8, in the versions of VGG19, ResNet152 and DenseNet201 respectively.
2. U-Net
3. SegNet

## Usage
1. Modify the **configuration.py** according to the hint in it.
2. Run **split_train_val.py**.
3. Set your model and training parameters in **train.py** and then run.

## Reference
1. I have referred to some nice repositories: [[1]](https://github.com/bodokaiser/piwise),
[[2]](https://github.com/ycszen/pytorch-ss)

## TODO
1. PSPNet
2. DeepLab
3. CRFAsRNN
4. And so on

