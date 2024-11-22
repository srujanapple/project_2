### Team Members

1. Satya Mani Srujan Dommeti  
2. Arjun Singh  
3. Akshitha Reddy Kuchipatla  
4. Vamsi Krishna  



# Model Implementation
This project includes the implementation of both linear regression and ridge regression. The models are designed to run concurrently, allowing the selection of the one that performs best based on the given dataset. To ensure robust performance assessment, k-fold cross-validation and bootstrapping techniques were applied. Additionally, the Akaike Information Criterion (AIC) was used to evaluate the quality and effectiveness of each model.

### K-fold Cross-Validation
We integrated k-fold cross-validation, providing users the option to set the value of *k* according to their needs. This customization enhances the evaluation process, enabling it to align with the unique characteristics of the dataset and ensuring a comprehensive and flexible analysis of the model's performance across various data partitions.

### Bootstraping
We implemented bootstrapping, allowing users to specify the number of iterations for generating resampled datasets. This approach involves repeatedly sampling the training data with replacement and fitting the model on these resampled datasets. By evaluating the model's performance, such as calculating mean squared error (MSE) for predictions on the test set, this method provides a robust way to estimate the stability and reliability of the model's performance under varying data distributions.
