# Assignment 5

Train a Model to classify Numbers from MNIST Data set in below Criteria: 
- Validation Accuracy 99.4%
- Parameters: < 10K
- Epochs: <= 15

Model is developed with the below architecture: 
- Max Pooling implemented twice alongwith 1 1X1 matrix to maintain the channel size
- Batch Normalization used to ensure the kernel values are within the range
- Dropout of 0.075 used to train the model, to tackle overfitting of the model and enable it to maintain good Validation Accuracy. The dropout is not added at first convolution to not drop any information at the first convolution level 
- Batch size of 128 is used, size 64 could also be used for similar accuracy
- Total Train loss is calculated and validated with Test Set
- The model parameters used are : 
- Epochs: 15
- Maximum Train Accuracy: % 
- Maximum Validation Accuracy: %

## Architecture Implemented for the Model
The code is divided in 5 Steps
- EVA4 - MNIST - First Step - Code Setup.ipynb
The first level working model with High Parameters, more Epochs, good Accuracy.

- EVA4 - MNIST - Second Step - Basic Skeleton.ipynb
The second step involves the basic skeleton of the model which shall be maintained for obtaining the desired result. This will serve as the foundation to fine tune. It has structured architecture, nearly less parameters, less epochs but may be slightly less accuracy. 

- EVA4 - MNIST - Third Step - BN and Regularization.ipynb
This step involes adding Batch normalization and Regularization using dropout to ensure the model has appropriate kernel values and is not overfitting respectively. This will ensure the model is learning appropriately to meet the desired result. 

- EVA4 - MNIST - Fourth Step - Adding GAP.ipynb
This step involves adding Global average pooling layer, to enable constant outcom with any input feature map dimensions. This allows the model to be flexible and could be optimized as required. 

- EVA4 - MNIST - Final Step - Capacity FC Augmentation.ipynb
The final step is to obtain the result by increasing the capacity i.e the number of parameters used, connecting a fully conected layer after GAP layer and applying image augmentation to Train and Test data set. This will ascertain optimised parameters for optimized resources used to get good accuracy. 