from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
import numpy as np
from pandas import read_csv

trainer = read_csv("C:/Users/Michael Huang/Documents/GitHub/SignML/sign_mnist_train/sign_mnist_train.csv")
#tester = read_csv("C:/Users/Michael Huang/Documents/GitHub/SignML/sign_mnist_test/sign_mnist_test.csv")

#print(trainer.head())
labels = np.unique(trainer["label"].values)
trainer.drop(["label"], axis=1) # 
