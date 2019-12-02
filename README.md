# Deep-learning
Multi-label image classification using various CNN architecture
Abstract: The main goal of this study is to build a deep neural network that can classify some
images into a set of classes 0-19. We have built a deep neural network with
ResNet50v2. We have managed to achieve 86.0298% accuracy on validation set which was run
on a google cloud compute engine n1-highmem-8 (8 vCPUs, 52 GB memory) with NVIDIA
Tesla V100 GPU.
Introduction
In this assignment we are aiming to perform a multi-label image classification task. Our main goal is to experiment with various deep learning architectures and techniques to achieve a high top1- accuracy. This is a challenging task as it is a multi-label image classification problem where each image contains multiple ground truth labels. This study aims to build a deep
neural network that can predict at least one label correctly.
It is important that the classifier has a reasonable accuracy and run time. The training and test
dataset contains 31925 and 15516 images respectively. The dataset has been labelled with 20
categories or classes. During training process various statistical and optimization methods has
been applied and evaluated in order to find optimal model for prediction of test dataset
observations.
This study is of significant importance for following reasons:
- In depth understanding of all current neural network algorithms and optimization methods
- Hone image recognition skill by exploring different CNN architectures
- Familiarity to tensorflow framework which acts as backbone of many deep learning APIs
Techniques
Data augmentation
Data augmentation involves generating new transformed versions of existing training data that
belong to the same class as the original training data. It is a popular technique to enhance the
overall performance of the network through expanding the training dataset using existing data and
helping to prevent overfitting.
Common transformations include flipping, rotation, scaling, shifting, cropping, introduction of
Gaussian noise, etc. In this project, images are randomly rotated in a range of 20 degrees,
randomly shifted horizontally and/or vertically to an extent of 20% of the original dimensions,
and randomly flipped horizontally and/or vertically. All the images in the dataset are processed by
this augmentation.
