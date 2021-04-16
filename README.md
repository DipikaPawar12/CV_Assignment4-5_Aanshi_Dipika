
<h1 align = "center">
    Homework Assignment 3 : Image Classification  
</h1>

---

<h1>Table of Content</h1>

- [Introduction](#introduction)
- [Approach](#Approach)
- [Results](#Results)
- [Platform](#Platform)
- [Installation guidelines](#Installation-guidelines)
- [References](#References)
- [Contributors](#Contributors)

## Introduction
---    
The human eye is capable of recognising, localising the features and classifying the images according to the variations present in the image at a very fast pace. In the man-made machines or systems, the capability of classifying the images according to the variations present is very less. The use of CNN in these systems help in increasing the classification accuracy as it creates a feature map for the features present within an image and predicts the label of the image using the minute-detailed as well as general features.

## Approach
---
- Dataset
The dataset is of caltech_birds 2010. The caltech-UCSD Birds 200(CUB-200) consists of 200 categories of bird images. There are total 6033 images with annotations containing bounding boxes and segmentation labels.

- Data Preprcoessing 
As the images are of different pixel ratio; in order to have easy in computation as well as to feed in the same pixel images to the model so that none of the features remains left out, the image pixels is made same. This is done by normalising the image where in every image is divided by 255, to keep the image pixels between 0 to 255. 
In the dataset, the image size is of about 180 X 180 due to which the system experiences much load, so the image size is reduced to a power of 2, here 64 X 64. But still due to heavy load on system, system crashes and so the size is reduced to power of 2, here 32 X 32 so that it can fit the model well.This size of image is further used.

- Data Augmentation
For a single image, augmentation in form of:
\begin{itemize}
    \item Flipping the image left to right or vice versa
    \item Adjusting the image brightness to 0.4
    \item Adjusting the image brightness to 0.2
    \item Cropping the image to 0.5
    \item Rotating the image several times by 90$^0$
\end{itemize}\\
After these many types of augmentation, for every image about 7 new images are generated. In total around $\boldsymbol{50,000}$ new images are generated and the dataset is expanded.

- Model Architecture


## Results
---
- Accuracy
The accuracy of the training dataset is about 99\% while the accuracy of the testing dataset is about 55\%. This accuracy is due to large number of classes(200) available for classification due to which it leads to high chances of misclassification which in turn which less accurate results.

- Entropy Loss Graph 
<img src="https://github.com/DipikaPawar12/CV_Assignment4-5_Aanshi_Dipika/blob/master/images/model_accuracy_loss_vs_epoch.jpeg" width="200" height="200" align="center">
This shows the curve depicting the relation between the training and testing loss with the number of epochs as well as the relation between the training and testing accuracy with the number of epochs.
 
- Learning Rate Graph
<img src="https://github.com/DipikaPawar12/CV_Assignment4-5_Aanshi_Dipika/blob/master/images/model_learning_rate.png" width="200" height="200" align="center">
 
- Momentum Graph
<img src="https://github.com/DipikaPawar12/CV_Assignment4-5_Aanshi_Dipika/blob/master/images/model_momentum.png" width="200" height="200" align="center">
The model with learning rate value of 0.1 or momentum value of 0.9 provides better results as compared to that model with learning rate and momentum combination which was even better compared to using the adam optimiser. The reason for choosing this values is that if the value chosen is too high then the model will learn quiclky and predicts the results quickly leading to misclassification and if value chosen is too low then the model may stop learning and may not be able to predict the results.

## Platform
---
- Google Colab


## Installation guidelines
---
- To clone this repository
 ```
git clone ___
 ```
- To install the requirements
```
pip install -r requirements.txt
```
- To install the dataset
```
pip install -q tfds-nightly tensorflow matplotlib
```

## References
---
- <a id="1">[1]</a> [Dataset](https://www.tensorflow.org/datasets/overview)
- <a id="2">[2]</a>  [Data Augmentation](https://www.tensorflow.org/tutorials/images/data_augmentation)
- <a id="3">[3]</a> [The Complete Beginner's Guide to Deep Learning: Convolutional Neural Networks](https://towardsdatascience.com/wtf-is-image-classification-8e78a8235acb)
- <a id="4">[4]</a> 
F. Sultana, A. Sufian, and P. Dutta
Advancements in Image Classification using Convolutional Neural Network
2018 Fourth International Conference on Research in Computational Intelligence and Communication Networks (ICRCICN).
- <a id="5">[5]</a> 
Dan C. Cires ̧an, Ueli Meier, Jonathan Masci, Luca M. Gambardella, and Jurgen Schmidhuber 
Flexible, High Performance Convolutional Neural Networks for Image Classification 
International Joint Conference on Artificial Intelligence.
- <a id="6">[6]</a> 
C.-C. Jay Kuo
Understanding Convolutional Neural Networks with A Mathematical Model.



## Contributors
---

| [Dipika Pawar](https://github.com/DipikaPawar12)                                                                                                            | [Aanshi Patwari](https://github.com/aanshi18)                                                                                                           
