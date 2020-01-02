from keras.models import Sequential, load_model
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.regularizers import l2

import sys

import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from time import sleep
import cv2


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
    for c, i in enumerate(images, 0):
        image = np.reshape(i, (28, 28))
        image = image.flatten()
        images[c] = np.array(image)

    return images, binary



def defineModel():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(x_test.shape[1:]), activation='relu', padding='same', activity_regularizer=l2(0.001)))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.2))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu',padding='same' ))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    #model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    

    #opt = SGD(lr=0.0008, momentum=0.9)
    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["categorical_accuracy"])
    return model


def summarize_diagnostics(history):
	# plot loss
	plt.subplot(211)
	plt.title('Cross Entropy Loss')
	plt.plot(history.history['loss'], color='blue', label='train')
	plt.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	plt.subplot(212)
	plt.title('Classification Accuracy')
	plt.plot(history.history['categorical_accuracy'], color='blue', label='train')
	plt.plot(history.history['val_categorical_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	plt.savefig(filename + '_plot.png')
	plt.close()


def testModel():
    model = load_model("my_model.hl5")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    print("Hello!, We'll give you 5 seconds to hold n ASL character before taking a picture.")
    sleep(5)

    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.waitKey(0)
    cropped = gray[0:480, 150:420].copy()
    cropped = cv2.resize(cropped, (28, 28), interpolation=cv2.INTER_AREA)
    frame = img_to_array(cropped)
    #Quick evaluation.
    #datagen = ImageDataGenerator()
    #datagen.fit(x_test)
    #_, acc = model.evaluate_generator(datagen.flow(x_test, y_test), steps=(len(x_test)), verbose=1)
    #print('> %.3f' % (acc * 100.0))

    #pic = input("Give me the extension:")
    #img = load_img("Ctrue.png", color_mode="grayscale", target_size=(28, 28)) - Direct predictions (load image directly)
    img = img_to_array(frame)
    img = img.flatten()

    img = np.reshape(img, (-1, 28, 28, 1)) #- Load me for predicting

    #img = np.reshape(img, (28, 28)) - Load me for showing the iamge
    #plt.imshow(img)
    #plt.show()

    img = img/255.0 # Normalizes the img
    maxAcc = 0.00
    maxIndex = 0
    pred = model.predict_classes(img)
    for i in range(len(pred)):
        prob = model.predict_proba(img)[0]
        prob = (prob[pred[i]]*100)
        if(maxAcc < prob):
            maxAcc = prob
            maxIndex = i
    print(pred[i])
    #cv2.release()
    cv2.destroyAllWindows()
    #print(test_test)
    


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

   
    choice = int(input("0 for Prediction Webcam, anything else for Compiling the Model: "))
    if(choice == 0):
        testModel()
    else:
        model = defineModel()
        #history =  model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=30, verbose=1, batch_size=128)
        #x_train2 = np.array(x_train, copy=True)
        #y_train2 = np.array(y_train, copy=True)

        datagen = ImageDataGenerator(featurewise_center=False, 
                                     samplewise_center=False,
                                     featurewise_std_normalization=False, 
                                     samplewise_std_normalization=False,
                                     zca_whitening=False, 
                                     width_shift_range=0.1,
                                     height_shift_range=0.1,
                                     horizontal_flip=True,
                                     #vertical_flip=True, 
                                     rotation_range=10)
        datagen.fit(x_train)

        datagenTest = ImageDataGenerator()
        datagen.fit(x_test)

        model = defineModel()
        history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=32), steps_per_epoch=len(x_train)/32, validation_data=(x_test, y_test), verbose=1, epochs=20)
       
        summarize_diagnostics(history)
        model.save("my_model.hl5")
        


