from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
import numpy as np
from pandas import read_csv

trainer = read_csv("C:/Users/Huang/Documents/GitHub/SignML/sign_mnist_train/sign_mnist_train.csv")
#tester = read_csv("C:/Users/Michael Huang/Documents/GitHub/SignML/sign_mnist_test/sign_mnist_test.csv")

#print(trainer.head())
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
#    break
print(images[0])