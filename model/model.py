########################################################################################################################
# File_name:            model.py                                                                                       #
# Creator:              Collect AI                                                                                     #
# Created:              Tuesday - August 7, 2018                                                                       #
# Last editor:          Moises Daniel Garcia Rojas                                                                     #
# Last modification:    Monday - August 20, 2018                                                                       #
# Description:          Definition of ImageClassifier class and its training and prediction methods.                   #
#                       train_new, save_model, open_model were added later for the purpose of the challenge            #
########################################################################################################################

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model
import h5py

from config.config import BATCH_SIZE, NUM_CLASSES, EPOCHS, IMG_ROWS, IMG_COLS, TRAIN_SIZE


class ImageClassifier:
    """
    Trains a simple convnet on the MNIST dataset.
    """
    def __init__(self):
        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, IMG_ROWS, IMG_COLS)
            x_test = x_test.reshape(x_test.shape[0], 1, IMG_ROWS, IMG_COLS)
            input_shape = (1, IMG_ROWS, IMG_COLS)
        else:
            x_train = x_train.reshape(x_train.shape[0], IMG_ROWS, IMG_COLS, 1)
            x_test = x_test.reshape(x_test.shape[0], IMG_ROWS, IMG_COLS, 1)
            input_shape = (IMG_ROWS, IMG_COLS, 1)

        # create a sequential convnet model in Keras
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(NUM_CLASSES, activation='softmax'))

        # define loss, optimizer and evaluation metric
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        self.model = model
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def train(self):
        """
        Train the model
        :return:
        """
        # convert types
        x_train = self.x_train.astype('float32')
        x_test = self.x_test.astype('float32')

        # scale
        x_train /= 255
        x_test /= 255

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(self.y_train, NUM_CLASSES)
        y_test = keras.utils.to_categorical(self.y_test, NUM_CLASSES)


        # finally fit the model on the data
        self.model.fit(x_train, y_train,
                       batch_size=BATCH_SIZE,
                       epochs=EPOCHS,
                       verbose=1,
                       validation_data=(x_test, y_test))
        return True

    def predict(self, newdata):
        """
        Predict class from the features
        :param newdata:
        :return:
        """
        return self.model.predict(newdata)

# New methods
    def train_new(self, labels, attributes):
        """
        Train the model
        :return:
        """
        # convert types
        # Format attributes
        x = attributes.reshape(labels.shape[0], IMG_ROWS, IMG_COLS, 1)
        # convert types
        x_train = x[0:int(TRAIN_SIZE * labels.shape[0])].astype('float32')
        x_test = x[int(TRAIN_SIZE * labels.shape[0]):].astype('float32')

        # scale
        x_train /= 255
        x_test /= 255

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(labels[0:int(TRAIN_SIZE * len(labels))], NUM_CLASSES)
        y_test = keras.utils.to_categorical(labels[int(TRAIN_SIZE * len(labels)):], NUM_CLASSES)
        # finally fit the model on the data
        self.model.fit(x_train, y_train,
                       batch_size=BATCH_SIZE,
                       epochs=EPOCHS,
                       verbose=1,
                       validation_data=(x_test, y_test))
        return True

    def save_model(self):
        self.model.save("my_model.h5")

    def open_model(self):
        # Load pre-trained keras model with the following configuration,
        # BATCH_SIZE = 512, NUM_CLASSES = 10, EPOCHS = 1
        self.model = load_model("my_model.h5")