from keras.models import Sequential, load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD

import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


trainer = read_csv("C:/Users/Huang/Documents/GitHub/SignML/sign_mnist_train/sign_mnist_train.csv")
labels = trainer["label"].values
trainer = trainer.drop(["label"], axis=1) #


tester = read_csv("C:/Users/Huang/Documents/GitHub/SignML/sign_mnist_test/sign_mnist_test.csv")
testlabels = tester["label"].values
tester = tester.drop(["label"], axis=1)



def preProcessing(stuff, classes):
    OH = OneHotEncoder(sparse=False)
    binary = classes.reshape(len(classes), 1)
    binary = OH.fit_transform(binary)   

    images = stuff.values
    #images = [images[i]/255.0 for i in range(images.shape[0])]
    for c, i in enumerate(images, 0):
        image = np.reshape(i, (28, 28))
        #plt.imshow(image)
        #plt.show()
        #break
        image = image.flatten()
        images[c] = np.array(image)

    return images, binary



def defineModel():
    model = Sequential()

    model.add(Conv2D(64, (3, 3), input_shape=(x_test.shape[1:]), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(BatchNormalization())

   # model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
   # model.add(Dropout(0.2))
   # model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(y_train.shape[1], activation='softmax'))
    model.add(Dropout(0.5))
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["categorical_accuracy"])
    return model


#def trainModel(model):


    #return model

if __name__ == "__main__": 
    data, labels = preProcessing(trainer, labels)
    x_train, x_test, y_train, y_test = train_test_split(data, labels,  test_size=0.33, random_state=42)

    # X contains a bunch of pixels, Y contains their respective labels.

    x_train = x_train.astype('float32')
    x_train = x_train/255.0
    x_train = np.reshape(x_train, (x_train.shape[0], 28, 28, 1))

    x_test = x_test.astype('float32')
    x_test = x_test/255.0
    x_test = np.reshape(x_test, (x_test.shape[0], 28, 28, 1))


   
   # model = defineModel()
    model = load_model("my_model.hl5")
    
    img = load_img("C:/Users/Huang/Documents/GitHub/SignML/TestC.jpg", color_mode="grayscale", target_size=(28, 28))
    img = img_to_array(img)
    img = img.flatten()
    #img = np.reshape(img, (-1, 28, 28, 1))
    img = np.reshape(img, (28, 28))
    #img = img/255.0
    plt.imshow(img)
    plt.show()

    #test = model.predict_classes(img)
    #print(test)
    #for i in range(3):
    #test_test = model.predict_proba(img)
    #test_test = "%.2f" % (test_test[test]*100)
    #print(test_test)
    
    #img = load_img("C:/Users/Michael Huang/Documents/GitHub/SignML/TestC.jpg", target_size=(28, 28))
    #img = img_to_array(img)
    #img = np.reshape(img, (-1, 28, 28, 1))

    #test = model.predict(img).round()
    #print((test))
    #print("Just a new line")
   # model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=40, verbose=1, batch_size=128)
   # testStuff, testlabels = preProcessing(tester, testlabels)
   # testStuff = testStuff.reshape(testStuff.shape[0], 28, 28, 1)
    #y_pred = model.predict(testStuff).round()
    #print(accuracy_score(testlabels, y_pred))
    #score = accuracy_score()
  #  model.save("my_model.hl5")

