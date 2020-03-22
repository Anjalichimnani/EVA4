# Assignment 9
## CIFAR10 Implementation using ResNet with Albumentations and GradCam

ResNet18 architecture implemented using Base Block and ResNet Block. The code is modular and all the components are parameterized for flexibility and maintenance. Transformations using Albumentations are performed on Train set and Transformations on Test set are simple. 
GradCam is implemented to understand the activations of the Model for different images. 

### Model Analysis: 
- Target: 
  - Architecture - [ResNet18!](https://arxiv.org/abs/1512.03385)
  - Albumentations - [Albumentations!](https://github.com/albumentations-team/albumentations)
  - Validation Accuracy >= 85%
  - Epochs - Any Number 

- Results:
  - Best Train Accuracy: 95.43%
  - Best Test Accuracy: 89.44%
 
Analysis:
  - Epochs used: 25
  - Total Parameters used: 11,173,962
  - Albumentation Transfor of HorizontalFlip, Rotate, HueSaturationValue, Cutout are applied along with Normalize and ToTensor. 
  - GradCam is implemented and the data is visualized using values. It culd be further enhanced for HeatMap.
  - After applying appropriate Transforms, the model over-fitting has reduced and Test Accuracy has increased.
  
### References:
[ResNet!](https://github.com/kuangliu/pytorch-cifar)
[GradCam!](https://github.com/kazuto1011/grad-cam-pytorch)

### Team Members: 
  - BhKPriyanka
  - Pratik Jain
  - Siddharth Surange  
  - Anjali Chimnani
