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
Data augmentation:
Data augmentation involves generating new transformed versions of existing training data that
belong to the same class as the original training data. It is a popular technique to enhance the
overall performance of the network through expanding the training dataset using existing data and
helping to prevent overfitting.
Common transformations include flipping, rotation, scaling, shifting, cropping, introduction of
Gaussian noise, etc. In this project, images are randomly rotated in a range of 20 degrees,
randomly shifted horizontally and/or vertically to an extent of 20% of the original dimensions,
and randomly flipped horizontally and/or vertically. All the images in the dataset are processed by
this augmentation.
Using pre-trained models for classification problems:
Two main challenges faced with training a deep neural network for classification questions are the
requirements of a large amount of training data and of significant computation power. A
solution to these challenges involves fine-tuning convolutional networks that have been
pre-trained on large scale image classification problems. This approach significantly outperforms
training from scratch as it provides the advantages of reduced number of parameters that need to
be tuned and overall reduce computation time. Some of the commonly used pre-trained models
include VGG, ResNet, MobileNet, Xception, etc.

VGG16
In this study we use a VGG16 model that has pre-trained on ImageNet dataset. VGG16 is a
16-layer convolutional neural network model proposed by K. Simonyan and A. Zisserman from
the University of Oxford in the paper “Very Deep Convolutional Networks for Large-Scale Image
Recognition”. The model achieves 92.7% top-5 test accuracy in ImageNet, which is a dataset of
over 14 million images belonging to 1000 classes. It was one of the famous model submitted to
ILSVRC-2014. It makes the improvement over AlexNet by replacing large kernel-sized filters (11
and 5 in the first and second convolutional layer, respectively) with multiple 3x3 kernel-sized
filters one after another. VGG16 was trained for weeks and was using NVIDIA Titan Black
GPU’s. This model consists of 16 convolutional layers and is very appealing because of its very
uniform architecture. VGG network is characterized by its simplicity, using only a few
convolutional layers stacked on top of each other in increasing depth. Reducing volume size is
handled by max pooling. Three fully-connected layers, each with 4,096 nodes are then followed
by a softmax classifier.
It is currently the most preferred choice in the community for extracting features from images.
The weight configuration of the VGGNet is publicly available and is being used in many other applications and challenges as a baseline feature extractor. However, VGGNet consists of 138
million parameters, which can be a bit challenging to handle. There are two major drawbacks
with VGGNet:
- It is painfully slow to train.
- The network architecture weights themselves are quite large (concerning disk/bandwidth).
Due to its depth and number of fully-connected nodes, VGG16 is over 533MB. This makes
deploying VGG a tiresome task.VGG16 is used in many deep learning image classification
problems; however, smaller network architectures are often more desirable. But it is a great
building block for learning purpose as it is easy to implement.

ResNet
Recently, the state-of-the-art CNN architecture is going deeper and deeper. However, increasing
network depth does not work by simply stacking layers together. Deep networks are hard to train
because of the notorious vanishing gradient problem—as the gradient is back-propagated to
earlier layers, repeated multiplication may make the gradient infinitely small. As a result, as the
network goes deeper, its performance gets saturated or even starts degrading rapidly[2].
Before ResNet, there had been several ways to deal the vanishing gradient issue, but none seemed
to really tackle the problem once and for all.

ResNet presents a residual learning framework to ease the training of networks that are
substantially deeper. Residual neural networks do this by utilizing skip connections, or shortcuts
to jump over some layers. Typical ResNet models are implemented with single-layer skips. One
motivation for skipping over layers is to avoid the problem of vanishing gradients, by reusing
activations from a previous layer until the adjacent layer learns its weights. During training, the
weights adapt to mute the upstream layer, and amplify the previously-skipped layer. Skipping
effectively simplifies the network, using fewer layers in the initial training stages. This speeds
learning by reducing the impact of vanishing gradients, as there are fewer layers to propagate
through. The network then gradually restores the skipped layers as it learns the feature space[12].
It provides comprehensive empirical evidence showing that these residual networks are easier to
optimize, and can gain accuracy from considerably increased depth.

InceptionResNetV2
Very deep convolutional networks have been central to the largest advances in image recognition
performance in recent years. One example is the Inception architecture that has been shown to
achieve very good performance at relatively low computational cost. Recently, the introduction of
residual connections in conjunction with a more traditional architecture has yielded
state-of-the-art performance in the 2015 ILSVRC challenge. Invention of this model proves that
training with residual connections accelerates the training of Inception networks significantly.
The network is 164 layers deep and can classify images into 1000 object categories, such as
keyboard, mouse, pencil, and many animals. As a result, the network has learned rich feature
representations for a wide range of images. The network has an image input size of 299*299.
With an ensemble of three residual and one Inception-v4, this network achieved 3.08 percent
top-5 error on the test set of the ImageNet classification (CLS) challenge.

Experiments and Results
In this section we describe the experiments we have carried out with three different models
namely VGG16, ResNet50 and ResNetInceptionV2. Initially we introduce the datasets used, our
implementation, the experiments conducted and the results achieved.

Labels   |Total images  |
 :-:     | :-:          |
 20      | 31925        |
 
 Table 1: Summary of data set used in our experiments
 
 Implementation
We implemented all of the deep CNN models used in our experiments in tensor.keras. We used
transfer learning techniques to train our model on pre-trained model weights on the ImageNet[6]
dataset. These weights were used as the initialisation weights. We fine tuned the weights for all of
the layers of each of the models. We used the Adam optimizer for model training with an initial
learning rate of 1e-5 for all the layers. The learning rate decay of learning/epochs was used,
where the number of epochs was set to 20. Early stopping and batch size of 64 were used in all
experiments. The input sizes were always kept (256, 256).

Experiment results
The following table summarises the accuracy results we achieved from running experiments on
the three different model architectures. We can see that the ResnetInceptionv2 model outperforms
all other models by far. We can also see that when the weights of all of the layers were frozen the
accuracy is not as high. However the accuracy improve drastically when the full model is trained
and the imagenet weights are only used as initial weights. Furthermore we also tuned the
hyperparameters. The only hyperparameters with significant effect were image size and learning
rate. We tried image sizes of (160, 160) and (256, 256). The following learning rates were also
experimented with. 1e-2, 1e-3, 1e-4, 1e-5, 1e-6. We found that 1e-5 produced the best results.

Model             |Accuracy  |
 :-:              | :-:      |
 ResnetInceptionV2| 86.0298% |
 Resnet50         | 70.7712% |
 VGG16            | 69.0368% |
 
 Table 2: Accuracy results from three different models when all of the layers were trained
 
 Model             |Accuracy |
 :-:              | :-:      |
 ResnetInceptionV2| 80.4326% |
 Resnet50         | 68.8637% |
 VGG16            | 66.8726% |
 
 Table 3: Accuracy results from three different models when of the layers were frozen
 
 !(https://github.com/Mimimkh/Deep-learning/blob/master/deep%20multi_label%20image%20classification.JPG)
 
 Discussion
In the course of this study we found that ResnetInceptionV2 outperforms other models such as
VGG16 and Resnet50 by far, this is due the fact that ResnetInception model combines techniques
from both Inception Network and residual networks, both of which have performed really well in
the exceeding state of the art single frame performance on the ImageNet validation dataset. When
using transfer learning, a pooling layer and dense layer can be added to the base model and the
pretrained weights of the model can be used, however tuning the layers of the model to our data
can significantly increase the accuracy especially if the data on which the base model was trained
on is similar to the training dataset. Something we didn’t try was training our model on different
data sources such as MS-COCO, NUS-WIDE and compare the results of based on initial weights
from those dataset. This is mainly due to the fact that MS-COCO pretrained models were mostly
built on caffe rather than tensorflow hence it required conversion of the weights from caffe to
probuff and then importing to tensorflow. We opted against it and concentrated on imagenet as the
only source of initial weights for our models. There is also room for further hyper parameter
tuning however we focused only on the learning rate as it has the highest impact on the accuracy
of the model.

Conclusion
In this study we investigated the various deep learning architectures namely vgg16, resnet50 and
Resnet Inception 2. We used transfer learning to train the models utilising the Imagenet weights
as initial weights. We found that training all of the layers achieves a higher a accuracy. We also
found that ResnetInceptionv2 achieves a higher accuracy compared with other architectures. This
is due to the fact that ResnetInceptionV2 combines features from two different techniques
Ineption networks and Residual networks.

Device and software specs
Tensorflow version: TensorFlow version is 1.13.1
Server specs: n1-highmem-8 (8 vCPUs, 52 GB memory)
GPU specs: Tesla V100-SXM2-16GB
