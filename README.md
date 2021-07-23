# BMI-classification-using-CNN
                                      METHODOLOGY:
Convolutional neural networks are a subfield under Deep learning which is used for the analysis of visual imagery. It is implemented as a classification technique that helped us to process and classify the images. There are also other classification techniques like the random forest, decision tree, etc. But we choose the CNN model as it uses a backpropagation technique which makes the process even more efficient and effective when compared to other algorithms considering the complexity of image data. The reliability of the model largely depends on the features extracted from the images.
We have applied various Data augmentation techniques such as rotation_range,width_shift_range,height_shift_range,shear_range,zoom_range,horizontal_flip,fill_mode which have been used to augment the size of the input dataset by applying position and color augmentation.
Image classification is a technique of segmenting images into different categories based on feature extraction. The system learns to do feature extraction and the main concept of CNN is, it uses convolution of image and various filters to generate invariant features which are passed on to the next layer. The features in the succeeding layer are convoluted with different filters to generate more invariant and abstract features and the process continues till we get the final feature/output which is invariant to occlusions. Features of an image involve pixel intensity, pixel values, corners, edges, regions of interest points, ridges, etc. The feature vectors are generated from the analysis of images, which served as a basis to distinctly extract or learn the features from the given image dataset without any human supervision. Since our data consists of categorical cross-entropy, we have made use of softmax activation function, Categorical cross-entropy loss function, and most prevailing and effective Adam optimizer with a learning rate of 0.00001. The resultant CNN model can take an input image and extract features and learns at learning rate provided through Adam optimizer such that one image of a categorical label can be easily distinguished from another label which makes the model computationally efficient.
                                                   
                                                   IMPLEMENTATION:
Deep Learning is becoming a very popular subset of machine learning due to its high level of performance and reliable results. A great method to use deep learning to classify images is to create a convolutional neural network (CNN). The Keras library in Python makes it pretty simple to make a CNN.

Loading data and data augmentation:
The first step we have implemented is to load the dataset which has been manually collected and organized from various resources throughout the internet. So, using the python library module keras we have imbibed the dataset, and using the Image data generator class we have performed eclectic data augmentation techniques to enlarge our input dataset. Also, we have divided the data into train and test sets and found out the labelled indices of different categories.

Model structure and layers:
The model type that we used is Sequential. Because it is one of the easiest ways to build a model in Keras. With help of add() function, we added different layers such as Maxpooling2D, Conv2D, Dropout, Flatten, and Dense layers.
Our first 2 layers are Conv2D and Maxpooling2D layers. These are known as convolutional and pooling layers which deal with the input images and are seen as 2-dimensional matrices. The convolution layer tries to extract higher-level features by replacing data for every (one) pixel with a worth computed from the pixels covered and Pooling layers reduce the spatial size of the output by replacing values within the kernel by a function of these values.
The term “dropout” refers to dropping out units (both hidden and visible) in a neural network which is used to get rid of the overfitting of the model.
In between the Conv2D layers and the dense layer, there is a Flatten layer which serves as a connection between the convolution and dense layers.
![model_summary](https://github.com/Krisshvamsi/Predictive-Analysis-of-BMI-using-captured-Image-Analysis-with-Covolutional-Neural-Network/blob/main/model_summary.PNG)
 
Activation functions:
The initial layers have been plugged in with an activation function ReLU which returns 0 if it receives any negative input, but for any positive value x, it returns that value.
The output layer has the activation function as Softmax which makes the output sum up to 1 so the output can be interpreted as probabilities. The model will then make its prediction supported which option has the very best probability.

Compiling the model:
We used the Adam optimizer with a learning rate of 0.00001 because Adam is a replacement optimization algorithm for stochastic gradient descent for training deep learning models. It combines the simplest properties of the AdaGrad and RMSProp algorithms to supply an optimization algorithm which will handle sparse gradients on noisy problems. We used categorical cross-entropy as a loss function because the prediction classes of our problem are mutually exclusive and it is the most common choice of classification. A lower score indicates that the model is performing better.
To make things even easier to interpret, we used the ‘accuracy’ metric to see the accuracy score on the validation set when we train the model.

Training the model:
Using the fit_generator() function we have trained the model with the following parameters: training data (train_X), target data (train_y), validation data, steps per epoch, and the number of epochs.
We have trained the model using 500 epochs with validation steps being 4 and steps_per_epoch being 16. After training the model for 500 epochs we got an accuracy of 82% approximately on our validation set. Finally, we have plotted a graph to understand the fitting of the model with a validation set and we observed a good fit plot (nearly coinciding).

![train_vs_validation accuracy](https://github.com/Krisshvamsi/Predictive-Analysis-of-BMI-using-captured-Image-Analysis-with-Covolutional-Neural-Network/blob/main/Final_metrics.PNG)

![classification report](https://github.com/Krisshvamsi/Predictive-Analysis-of-BMI-using-captured-Image-Analysis-with-Covolutional-Neural-Network/blob/main/Classification_report.PNG)

To understand the model even better with help of evaluation metrics we found out the confusion matrix and classification report of our model with 
Accuracy:0.8113207547169812,
Sensitivity:0.9285714285714286,
Specificity:0.8823529411764706

                                      FUTURE SCOPE AND CHALLENGES:
The model which we implemented requires an image as an input for the classifying it into a specific category. Also in future, we can further develop our deep learning model by integrating it with different prevailing technologies such as Internet of things (IoT), cloud computing, Computer vision, etc. Using IoT we can make our model more reliable by varying our dataset through image and also make it user interactive with help of IoT enabled devices such as cameras, sensors, etc. We can also build recommendation systems based on the classified output to help user with diet plans, nutrition and diagnosing the serious illness causing problems due to obese or underweight conditions. Also using openCV we can build a user interactive and an efficient model where users can directly upload an image through camera and the output can be displayed on the camera window itself. 
The primary challenge for deep learning is lack of dataset. This requires data acquisition or help from domain experts. Although there are techniques which can build a model on smaller datasets, the deep learning model requires larger datasets to create a model with better accuracy, and to avoid overfitting, but unfortunately creating large input datasets is a time taking process. 
Secondly, developing a deep learning model from the given input images by feature extraction and learning on its own can be a time-consuming process.

                                              CONCLUSION:
Convolutional neural networks is very widely used in various domains like medical analysis. It has become a prominent method in analysing the complex data for image classification. Being known the challenges and limitations present in the CNN can help in developing the model further. BMI classification is a deep learning model which classify the input image of humans into specified categories. Therefore, based on diagnosis of BMI through image classification one can easily follow corresponding diet procedures and can maintain a healthy BMI, also making it easy and supportive for the nutritionists for analysing and diagnosing patient’s health status.
