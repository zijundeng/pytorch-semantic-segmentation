## Preparation
1. Download *leftImg8bit_trainvaltest.rar* and *gtFine_trainvaltest.rar*. 
2. Extract them and put in a directory (called *root*).
3. Set the path of *root* in *config.py*.
4. Run *preprocess.py* to generate suitable format of data for *cityscapes dataset*.

## Usage
```python
import ...
from datasets.cityscapes import CityScapes
from torch.utils.data import DataLoader
train_set = CityScapes('train', simul_transform=train_simul_transform, transform=train_transform, target_transform=MaskToTensor())
train_loader = DataLoader(train_set, batch_size=training_batch_size, num_workers=16, shuffle=True)
```

## TODO
1. Support another coarse part of dataset.