from keras.models import Sequential, load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
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

    model.add(Conv2D(32, (3, 3), input_shape=(x_test.shape[1:]), activation='relu', padding='same', activity_regularizer=l2(0.001)))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu',padding='same' ))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    #model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))
    #model.add(Dropout(0.2))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    

    opt = SGD(lr=0.0008, momentum=0.9)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["categorical_accuracy"])
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
    #folder = "C:/Users/Huang/Documents/GitHub/SignML/"
    
    print(model.evaluate(x_test, y_test))
    #pic = input("Give me the extension:")
    img = load_img("C:/Users/Huang/Documents/GitHub/SignML/a.jpg", color_mode="grayscale", target_size=(28, 28))
    img = img_to_array(img)
    img = img.flatten()
    img = np.reshape(img, (-1, 28, 28, 1))
    #img = np.reshape(img, (28, 28))
    img = img/255.0
    #plt.imshow(img)
    #plt.show()

    test = model.predict_classes(img)
    print(test)
    #for i in range(3):
    test_test = model.predict_proba(img)[0]
    test_test = "%.2f" % (test_test[test]*100)
    print(test_test)
    


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

    #testModel()
    choice = int(input("Load or Define? (0, 1)"))
    if(choice == 0):
        #model = load_model("my_model.hl5")
        testModel()
    else:
        model = defineModel()
        history =  model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=30, verbose=1, batch_size=128)
        summarize_diagnostics(history)

        testStuff, testlabels = preProcessing(tester, testlabels)
        testStuff = testStuff.reshape(testStuff.shape[0], 28, 28, 1)
        y_pred = model.predict(testStuff).round()
        print(accuracy_score(testlabels, y_pred))
    #score = accuracy_score()
        model.save("my_model.hl5")

