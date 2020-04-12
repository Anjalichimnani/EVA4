# Assignment 11
## CIFAR10 Implementation using Customized Network with ResBlock, Albumentations and LR Finder, Super Convergence using One Cycle Policy

Custom Network implemented using convolution blockas along with ResBlocks The code is modular and all the components are parameterized for flexibility and maintenance. Transformations using Albumentations are performed on Train set and Transformations on Test set are simple. 
LR Finder is implemented using reference from [LRFinder!](https://github.com/davidtvs/pytorch-lr-finder) with SGD Optimizer with Momentum. 
The schedule One Cycle Policy is used to train the model to Max_LR = 0.006 within 5 epochs. Training schedule for cyclic policy is plotted. 

### Model Analysis: 
- Target: 
  - Architecture - [ResNet18!](https://arxiv.org/abs/1512.03385)
  - Albumentations - [Albumentations!](https://github.com/albumentations-team/albumentations)
  - Validation Accuracy >= 90%
  - Epochs - 24 

- Results:
  - Best Train Accuracy: > 98.67%
  - Best Test Accuracy: > 91%
 
Analysis:
  - Epochs used: 24
  - Total Parameters used: 6,573,120
  - Albumentation Transfor of Padding, RandomCrop(32, 32), FlipLR, Cutout are applied along with Normalize and ToTensor. 
  - Maxmimum LR of 0.006 is used based on the outcome from LR finder. 

### Training Schedule:
[Cyclic Training Schedule!](https://github.com/Anjalichimnani/EVA4/blob/master/Assignment_11/Graphs/Training_Schedule.PNG)
  
### References:
[ResNet!](https://github.com/kuangliu/pytorch-cifar)
[LRFinder!](https://github.com/davidtvs/pytorch-lr-finder)
