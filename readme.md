# Weather Image Recognition using Convolution Neural Networks
Technologies used: Python, PyTorch...

## What is the goal?
The goal of this project is to train a Convolution Neural Network (CNN) to identify various weather conditions from images. This allows the application of knowledge and techniques learned in this course to a real-world problem with an actionable dataset from a reputable data source. 

### What can this do for us?
In the right conditions, training such a CNN can lead to numerous significant impacts to meteorology and the current understanding of our climate. 

## Why Neural Networks?
Neural Networks can discern subtle patterns not easily noticed by other algorithms, allowing us to have a new lens on an old problem and gain fresh insights or new developments in meteorology as a whole. 

### Why CNNs?
CNNs are a type of neural network that are particularly good at image recognition. They  are able to identify patterns in images by looking at small parts of the image at a time.  They then combine the results of these small parts to make a decision.

Imagine you are trying to identify a cat in an image. You would look at the image and notice that there are two eyes, a nose, and a mouth. You would then combine these features to make a decision that the image is of a cat. CNNs work in a similar way. They look at small parts of the image and combine the results to make a decision.

### Issues with CNNs...
Given the example above, you can understand how this can lead to misclassification and other issues. Consider the example again. If we were to provide an image of a dog with the notable features of a nose and two eyes. The CNN would still classify the image as a cat since the features are similar. The CNN does not know how to distinguish between a cat nose and a dog nose. 

## Dataset and other information...
The dataset that will be used throughout this project will be the ‘Weather Image Recognition’ dataset from Kaggle, published by Jehan Bhathena. 
The URL for this dataset is: https://www.kaggle.com/datasets/jehanbhathena/weather-dataset/data. 

The images are sourced from the Harvard Dataverse where it was published by Haixia Xiao of the Nanjing Joint Institute for Atmospheric Sciences. 
The URL to the Harvard Dataverse dataset is: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/M8JQCR.