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



### Overview of Ideas in Related Works

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


## My Investigation

What my work mainly revolves around is Transfer Learning, that is I loaded pre-trained models, which were built for 
some specific image classification problem, like, for example, classifying flowers or animals in general, and I use part of 
the structure of that loaded model and restructure it a little so that it can be trained to solve my, dog breed classification, problem;
*restructure* in Transfer Learning implies removing some of the layers (the easiest to remove are generally the 'external' ones, 
such as input layer or output layer, or some dense layers, and adding one's own layers that fit the inputs' and outputs' shapes.

For this project I have used `tensorflow` library, and can immediately stress its ease of use. It has many pre-trained models,
out of which I decided to work with the following list: ResNet50, VGG16, AlexNet. 
I have also tried implementing my own model, but got stuck with non-convergence, however, got pretty interesting results and 
gained much insight along the way. 

It important to mention that the fine-tuning that I did and which I uploaded to github was mainly replacing the last `Dense` layer 
of the original pre-trained model, with the `Dense` layer with output shape equals to the number of classes in my case.
Also it involved replacing the `InputLayer` with one, specified for my input images.

I have mainly considered a pair of possible input shapes: (64, 64, 3) and (100, 100, 3), as I anyway needed to scale my data down to constant shape,
although the above shapes are quite small resolution-wise, meaning that some data could have been missed.

As an optimizer, I chose the RMSProp with default parameter of `learning_rate`, set to 0.001, and `rho`, set to 0.9.


### Top 10 most frequent classes
I decided to start off with the smaller dataset of images which i used for training models;
This dataset is smaller in terms of having far less breeds, keeping only 10 most frequent breeds.
The train data was split into 90/10 as train/test, which I assumed to be a good idea, given a relatively low number of data points.
The **main metric** that I used for assessing the performance of models is accuracy, for all models I specified loss function as `'categorical_crossentropy'`

#### Fine-tuned AlexNet :

This model turned out to be the worst of all I have implemented. Comparing its loss history plot, which was built on 150 epochs

![AlexNet train loss](https://user-images.githubusercontent.com/61096766/149646047-bc689d45-bb84-4b2f-9b0a-f5cd84a43e4d.png)

and the train accuracies history plot: 

![AlexNet train accuracy](https://user-images.githubusercontent.com/61096766/149646048-b36b278a-6c06-4c9a-9bc5-70b722eeb5e4.png)


to the history plots of

#### Fine-tuned VGG16:

for loss:

![VGG16](https://user-images.githubusercontent.com/61096766/149646033-67cec24c-a4dd-4c42-870c-b9cc3e601125.png)

And for accuracy:

![VGG16 accuracy](https://user-images.githubusercontent.com/61096766/149646036-533951c9-e4f3-4ed3-9ff2-124aca772377.png)

VGG16 looks much more smoother and superior to AlexNet, and smoothness of lines might also be related to the 
test accuracies, as, although both models *overfitted*, at least VGG16 did much better, with train-test accuracy margin of just 0.3,
while AlexNet produced 0.7 accuracies during training, but only 0.1 accuracy for the test data (test data - 0.1 of the split of the top 10
dataset). Therefore, I decided not to move the baseline model (AlexNet) to extensive original dataset with 120 classes.

### For entire dataset

Here I decided to cut the epochs a little bit, in hopes to prevent overfitting, but it did not help much.
Let's now compare the other 2 models, being *ResNet50* and *VGG16*

#### Fine-tuned ResNet50

Loss history plot:

![ResNet50 for all dataset history of loss values](https://user-images.githubusercontent.com/61096766/149647139-2dd7b579-ef36-4107-ac22-90d170cb694b.png)

Accuracy history plot:

![ResNet50 for all dataset history of accuracy values](https://user-images.githubusercontent.com/61096766/149647138-cd9baae9-a7c6-4074-91b7-1d973bb9700e.png)

#### Fine-tuned VGG 16

![VGG16 for all dataset history of loss values](https://user-images.githubusercontent.com/61096766/149647146-3e5feb96-ddca-42dc-99a5-ec3663dba3e9.png)

![VGG16_for_all_dataset history of accuracy values](https://user-images.githubusercontent.com/61096766/149647145-cd420ba9-474c-45c5-89dd-806cc7022a24.png)





## Conclusions and Further Research

For this project I had a bit of a time constraint, as models with millions of parameters take much more time to train, than what I
expected, and another slightly unpleasant insight has been the fact that I could not use GPU's on my computer.

However, the only thing I sincerely regret not trying is the data augmentation technique as I have seen multiple related works, 
containing this method, although not having much proven success, as it was used for training only, and the model later generated
predictions as a competition submission.

Another insight that I would like to write about is the overfitting, the fine-tuned models were just great on the training stage, with non-decreasing
and non-increasing accuracy and loss trends respectively.

