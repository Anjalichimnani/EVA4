# Assignment 8
## CIFAR10 Implementation using ResNet

ResNet18 architecture implemented using Base Block and ResNet Block. The code is modular and all the components are parameterized for flexibility and maintenance. 

### Model Analysis: 
- Target: 
  - Architecture - [ResNet18!](https://arxiv.org/abs/1512.03385)
  - Validation Accuracy >= 85%
  - Epochs - Any Number 

- Results:
  - Best Train Accuracy: 99.58
  - Best Test Accuracy: 87.73
 
Analysis:
  - Epochs used: 25
  - Total Parameters used: 11,173,962
  - Data Augmentation of RandomRotation and ColorJitter Applied. 
  - L2 Regularization applied. 
  - The model has very high Train accuracy, it could be further aligned (not overfitted) w.r.t Test accuracy by different methods of L1 regularization and dropout. 
  
### References:
[ResNet!](https://github.com/kuangliu/pytorch-cifar)