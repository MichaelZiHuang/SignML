from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.compose import ColumnTransformer

trainer = read_csv("C:/Users/Michael Huang/Documents/GitHub/SignML/sign_mnist_train/sign_mnist_train.csv")
labels = (trainer["label"].values)

trainer = trainer.drop(["label"], axis=1) #
#tester = read_csv("C:/Users/Michael Huang/Documents/GitHub/SignML/sign_mnist_test/sign_mnist_test.csv")
#CT = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder="passthrough")
#dataset = np.array(CT.fit_transform(dataset), dtype = int)
#print(dataset)
#print(onehot.categories_)


def preProcessing():
    OH = OneHotEncoder(sparse=False)
    binary = labels.reshape(len(labels), 1)
    binary = OH.fit_transform(binary)   

    images = trainer.values
    #images = [images[i]/255.0 for i in range(images.shape[0])]
    for c, i in enumerate(images, 0):
        image = np.reshape(i, (28, 28))
        image = image.flatten()
        images[c] = np.array(image)

    return images, binary



def defineModel():
    model = Sequential()

    model.add(Conv2D(16, (3, 3), input_shape=(x_test.shape[1:]), activation='relu', padding='same'))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(y_train.shape[1], activation='softmax'))

    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])
    return model

  

data, trash = preProcessing()
LB = LabelBinarizer()
labels = LB.fit_transform(labels)
print(labels)
x_train, x_test, y_train, y_test = train_test_split(data, labels,  test_size=0.33, random_state=42)

# X contains a bunch of pixels, Y contains their respective labels.

x_train = x_train.astype('float32')
x_train = x_train/255.0
x_train = np.reshape(x_train, (x_train.shape[0], 28, 28, 1))

x_test = x_test.astype('float32')
x_test = x_test/255.0
x_test = np.reshape(x_test, (x_test.shape[0], 28, 28, 1))



print(y_train)
#y_train = np_utils.to_categorical(y_train)
#print(y_train)
#y_test = np_utils.to_categorical(y_test)
#y_train = np_utils.to_categorical(y_train)
#y_test = np_utils.to_categorical(y_test)

class_num = 24

#print(y_train.shape)
model = defineModel()

#Conv2d Format: (batch, rows, cols, color channels)

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=25, batch_size=120)

score = model.evaluate(x_test, y_test, verbose=1)

print(score)
