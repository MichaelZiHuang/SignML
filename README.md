# SignML
A machine learning project based on interpreting letters, then phrases of sign language. This project uses the Kaggle Sign MNIST Database. Truth be told, I wanted to see if I create my own dataset, but creating a high quality dataset for a relatively short project wasn't feasible. I'm actually going to try and document this process instead of haphazardly reviewing my code like last time. So let's hope for the best! (Maybe I'll create my own as an expansion)

<h1>Day 1</h1>
<img src="amer_sign2.png">
So here's the picture that Kaggle provides. This (probably) represents the 26 letters the dataset contains. If we open up the training set, we see that the training set is very long, several scrolls long, so we have a lot of data. I'm running this on my local machine, so it may be a bit worrisome for some epoch testing. 

<pre><code>
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
import numpy as np
from pandas import read_csv

trainer = read_csv("C:/Users/Michael Huang/Documents/GitHub/SignML/sign_mnist_train/sign_mnist_train.csv")
#tester = read_csv("C:/Users/Michael Huang/Documents/GitHub/SignML/sign_mnist_test/sign_mnist_test.csv")

print(trainer.head())
<pre></code>

Some familiar imports. The first 3 are some things I'm more than likely going to use. That said, the new one is pandas. Kaggle gave us CSV files, it's only natural that we want to read said CSV files. The head prints out the literal information of the CSV file. Honestly, Pandas Dataframes are just interesting, and also something I'm first experiencing, so let's see how this goes. 

<pre><code>
labels = np.unique(trainer["label"].values)
trainer = trainer.drop(["label"], axis=1) #

images = trainer.values
#print(images.shape)
images = [images[i]/255.0 for i in range(images.shape[0])]
#print(img[0])
#/tiestarray = [info for _, info in img.iterrows()]
for c, i in enumerate(images, 0):
    image = np.reshape(i, (28, 28))
    image = image.flatten()
    images[c] = np.array(image)
</pre></code>

This took a surprisingly long amount of time. This is the first time I've tried to do a Kaggle dataset with a .csv, so I forgot that I needed to preprocess the data. So, this is what it does. It takes the .csv data, one hots them, then reshapes them into a usable 28x28 format instead of the original 1x784. It then flattens it. Truth be told, this flatten mechanic was advised since I have static data that I always expect to be that size, flattening now instead of doing it between each layer may help. The things I want to keep track here during future coding is as follows
<ul>
    <li> <code>images = [images[i]/255.0 for i in range(images.shape[0])]</code>
    <li> <code>image = image.flatten() </code>
</ul>

I want to know if flattening and one-hoting now makes a difference.