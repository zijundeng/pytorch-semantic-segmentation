# PyTorch for Semantic Segmentation
This repository contains some models for semantic segmentation and the pipeline of training and testing models, 
implemented in PyTorch.

## Models
1. Vanilla FCN: FCN32, FCN16, FCN8, in the versions of VGG, ResNet and DenseNet respectively. 
([Fully convolutional networks for semantic segmentation](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf))
2. U-Net ([U-net: Convolutional networks for biomedical image segmentation](https://arxiv.org/pdf/1505.04597))
3. SegNet ([Segnet: A deep convolutional encoder-decoder architecture for image segmentation](https://arxiv.org/pdf/1511.00561))
4. PSPNet ([Pyramid scene parsing network](https://arxiv.org/pdf/1612.01105))
5. GCN ([Large Kernel Matters](https://arxiv.org/pdf/1703.02719))
6. DUC, HDC ([understanding convolution for semantic segmentation](https://arxiv.org/pdf/1702.08502.pdf))
7. Deformable Convolution Network (in PSPNet version) ([Deformable Convolutional Networks](https://arxiv.org/pdf/1703.06211))

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
4. More dataset (e.g. ADE)
