from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import cv2
from imutils import paths
import os.path
import pickle

'''
Train model code
'''


def train_model():
    letter_folder = 'LettersCaptchas'

    # creating empty lists for storing image data and labels
    data = []
    labels = []
    for image in paths.list_images(letter_folder):
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (35, 35))

        # adding a 3rd dimension to the image
        img = np.expand_dims(img, axis=2)

        # grabing the name of the letter based on the folder it is present in
        label = image.split(os.path.sep)[-2]

        # appending to the empty lists
        data.append(img)
        labels.append(label)

    # converting data and labels to np array
    data = np.array(data, dtype="float")
    labels = np.array(labels)

    # scaling the values of  data between 0 and 1
    data = data / 255.0

    # Split the training data into separate train and test sets
    (train_x, val_x, train_y, val_y) = train_test_split(data, labels, test_size=0.3, random_state=13)

    # one hot encoding
    lb = LabelBinarizer().fit(train_y)
    train_y = lb.transform(train_y)
    val_y = lb.transform(val_y)

    # building model
    model = Sequential()
    model.add(Conv2D(20, (5, 5), padding="same", input_shape=(35, 35, 1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(26, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # using early stoping for avoiding overfitting
    estop = EarlyStopping(patience=10, mode='min', min_delta=0.001, monitor='val_loss')

    model.fit(train_x, train_y, validation_data=(val_x, val_y), batch_size=20, epochs=500, verbose=1, callbacks=[estop])

    # save the model to disk
    filename = 'trained_model.obj'
    pickle.dump(model, open(filename, 'wb'))

    filename_sec = 'data_lb.obj'
    pickle.dump(data, open(filename_sec, 'wb'))

    return
