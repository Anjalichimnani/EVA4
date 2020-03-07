# Assignment 7
## CIFAR10 Net

- The CIFAR10 Model has architecture with 3 convolution blocks and 3 Transition Blocks (MaxPooling), modeled to run on GPU. 
- It has one convolution block with Dilation of 2 to increase the Receptive Field of the Layer.  
- And another convolution block with 2 depthwise separable convolutions to optimize the parameters used to train the network. 
- The total receptive field is more than 44, it is nearly 89. 
- The model has Global Average Pooling (GAP) followed by Fully connected layer as the last layers to identify the 10 classes for input data set. 

### Model Analysis: 
- Target: 
  - Architecture C1C2C3C40 (3 MP)
  - One layer with Dilated Convolution
  - One of the layers use Depthwise Separable Convolution
  - GAP compulsory, add FC after GAP to target #of classes (optional)
  - Total Parameters < 1M, any number of Epochs

- Results:
  - Best Train Accuracy: 86.45
  - Best Test Accuracy: 83.59
 
Analysis:
  - Epochs used: 20
  - Total Parameters used: 155k
  - Data Augmentation of Random Horizontal Flip can be applied
  - Model aims to include dilation convolution and depthwise convolution.
  - Regularization can be applied. Further, Test data can be checked for appropriate cllasification and then appropriate architecture can be enhanced. 
