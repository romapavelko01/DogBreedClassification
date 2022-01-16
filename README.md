# Dog Breed Classification
Project for AI course at UCU

**Important:** in case one needs the data folder that I did not provide, 
it can be found [here](https://www.kaggle.com/c/dog-breed-identification/data)


## Introduction

### Idea description

There are many resources on dog breeds, what are their descent, how to take proper care of each of them. 
However, often, people are still unaware of what dog they meet on the streets and it leads to many inconveniences, 
some of them being just a daily life, like putting oneself to shame in front of their friends – die-hard dog-lovers; 
and some of them having criminally absurd origin, like giving wrong evidence to police officers who are in search 
of a blind man, who was always walking with his dog and who also got lost a week ago.


### Data/Environment/Background

There was a competition for most accurate dog breed classification model on Kaggle in 2017, so there was collected 
relatively a lot of data (why *relatively*? I will mention it in the **conclusions**) for different dogs photos 
and corresponding breeds. 

More specifically, there is a dataset (~750MB) of  20.6k images (of format .jpg) of 120 different dog breeds.



### Overview of ideas in related works

Most of the methods which I found on web use CNN’s [although Standford also considered K-Nearest Neighbors, 
SVM and Logistic Regression models, but which reaches seemingly low accuracy - ~48%, which one may consider inferior to random guess,
but one should also consider that there are totally 120 classes, so a random guess would lead to 1/120 ~ 0.8% accuracy], 
which is a type of neural networks most suitable for dealing with image data due to the specificity of its layers, which process
image applying kernels to the matrix of pixels - that is, applying convolution; this leads to extraction of features - 
high-level ones or low-level ones, which heavily depends on the parameters set for the kernel.
Pooling layers help reduce dimensionality. Most common image classification CNN's, such as AlexNet, ResNet have extremely 
complicated architectures, with tens of millions of parameters. However, what makes them similar is that there are multiple 
Convolutional layers inside the models, which is really helpful in detecting as many patterns in images as possible.


## EDA

So, as was mentioned there is a folder with train and test photos, but only the train labels .csv file is provided. Again, this is 
done on purpose, since it is a Kaggle competition and participants have to submit their own predictions over test images.

Train labels file contains 2 columns - *[id, breed]*, where *id* is just a filename of the image, with breed - no-spaced string corresponding 
to the breed of the dog on the id's image. Train and Test folders contain 10222 and ~10400 files. Each image has `.jpg` extension.

Now, let's check the distribution of breeds in the train dataset:

![Mostly, non-drastic](https://user-images.githubusercontent.com/61096766/149646046-71d10e79-6f46-4fd9-ac53-fdf97ac5bbdd.png)


We can see that it is smooth, there are no drastic peaks, but since there are so many breeds we cannot even say which bar reflects 
what breed frequency, since if they were labelled, it would still be unclear as there would be just too much text.


So, let's also see what the labelled distribution of the top 10 breeds is:

![Same, but labelled](https://user-images.githubusercontent.com/61096766/149646045-f05b703b-b610-4204-b74f-704867a4458c.png)


Good, let us see what the randomly selected 12 images look like:

![Random 12 doggies](https://user-images.githubusercontent.com/61096766/149646044-f2fcd832-6e30-4b3b-99cd-804a1cd5723c.png)

We can see right away that images are not of the same shape, although all of them are 3-channel (RGB), it might be an issue, as models want
inputs with equal shape, but with some preprocessing it might not even be an issue.

Let's finish off the EDA part with the top 10 most frequent breeds, which total to 1141 instances, and here are the examples, 
what these dogs look like:

![10 Dogs, most probable to see](https://user-images.githubusercontent.com/61096766/149646038-ce8c954a-4434-4c8f-9869-d82bd8f1692e.png)


### My investigation

What my work mainly revolves around is Transfer Learning, that is I loaded pre-trained models, which were built for 
some specific image classification problem, like, for example, classifying flowers or animals in general, and I use part of 
the structure of that loaded model and restructure it a little so that it can be trained to solve my, dog breed classification, problem;
*restructure* in Transfer Learning implies removing some of the layers (the easiest to remove are generally the 'external' ones, 
such as input layer or output layer, or some dense layers, and adding one's own layers that fit the inputs' and outputs' shapes.

For this project I have used `tensorflow` library, and can immediately stress its ease of use. It has many pre-trained models,
out of which I decided to work with the following list: ResNet50, VGG16, AlexNet. 
I have also tried implementing my own model, but got stuck with non-convergence, however, got pretty interesting results and 
gained much insight along the way. 


The models which I chose for fine-tuning 
