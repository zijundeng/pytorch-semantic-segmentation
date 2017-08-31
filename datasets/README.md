# Dataset

## PASCAL VOC 2012
1. Visit [this](https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/data/pascal), download SBD and
PASCAL VOC 2012
2. Extract them, you will get *benchmark_RELEASE* and *VOCdevkit* folders.
3. Add file *seg11valid.txt* ([download](
https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/data/pascal/seg11valid.txt))
into *VOCdevkit/VOC2012/ImageSets/Segmentation*
4. Put the *benchmark_RELEASE* and *VOCdevkit* folders in a folder called *VOC*
5. Set the path (*root*) of *VOC* folder in the last step in *voc.py*

## Cityscapes
1. Download *leftImg8bit_trainvaltest*, *gtFine_trainvaltest*, *leftImg8bit_trainextra*, and *gtCoarse* from the cityscapes website
2. Extract and put them in a folder called *cityscapes*
3. Set the path (*root*) of *cityscapes* folder in the last step in *cityscapes.py*
