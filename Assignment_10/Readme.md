# Assignment 10
## CIFAR10 Implementation using ResNet with Albumentations and LR Finder, ReduceLROnPlateau

ResNet18 architecture implemented using Base Block and ResNet Block. The code is modular and all the components are parameterized for flexibility and maintenance. Transformations using Albumentations are performed on Train set and Transformations on Test set are simple. 
LR Finder is implemented using reference from [LRFinder!](https://github.com/davidtvs/pytorch-lr-finder) with SGD Optimizer with Momentum. 
Reduce LR on Loss Plateau, [ReduceLROnPlateau!](https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau), is implemented with Learning Rate decay of Factor 0.1 and 0 patience. The steps taken can be seen by logs as: 

Log:
<Epoch     6: reducing learning rate of group 0 to 9.0000e-04.>

### Model Analysis: 
- Target: 
  - Architecture - [ResNet18!](https://arxiv.org/abs/1512.03385)
  - Albumentations - [Albumentations!](https://github.com/albumentations-team/albumentations)
  - Validation Accuracy >= 88%
  - Epochs - 50 

- Results:
  - Best Train Accuracy: >95%
  - Best Test Accuracy: > 90%
 
Analysis:
  - Epochs used: 50
  - Total Parameters used: 11,173,962
  - Albumentation Transfor of HorizontalFlip, Rotate, HueSaturationValue, Cutout are applied along with Normalize and ToTensor. 
  - Optimized LR of 0.009 is used based on the iutcome from LR finder. 
  - ReduceLROnPlateau helps to ensure the LR updates during loss plateau. 
  
### References:
[ResNet!](https://github.com/kuangliu/pytorch-cifar)
[LRFinder!](https://github.com/davidtvs/pytorch-lr-finder)
[ReduceLROnPlateau!](https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau)