# Assignment 4

Train a Model to classify Numbers from MNIST Data set in below Criteria: 
- Validation Accuracy 99.4%
- Parameters: < 20K
- Epochs: < 20

Model is developed with the below architecture: 
- Max Pooling implemented twice alongwith 1 1X1 matrix to maintain the channel size
- Batch Normalization used to ensure the kernel values are within the range
- Dropout of 0.075 used to train the model, to tackle overfitting of the model and enable it to maintain good Validation Accuracy. The dropout is removed at 
- Batch size of 125 is used, size 64 could also be used for similar accuracy
- Total Train loss is calculated and validated with Test Set
- The model parameters used are : 15358
- Epochs: 19
- Maximum Train Accuracy: 99.17 
- Maximum Validation Accuracy: 99.4%

![Architecture Image] (https://github.com/Anjalichimnani/EVA4/blob/master/Assignment_4/Architecture.jpeg)