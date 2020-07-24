#!/usr/bin/env python
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from sklearn.model_selection import train_test_split
from matplotlib.backends.backend_pdf import PdfPages
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

# from tensorflow.keras.applications.resnet50 import ResNet50


pickle_file = "/media/miro/WD/jetbot_obstacle_avoidance/data_224.pckl"
test_size = 0.35


def model_224(n_filter_out, print_summary=False):
    model = Sequential()

    model.add(Conv2D(filters=16, kernel_size=(4, 4), input_shape=(224, 224, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=32, kernel_size=(4, 4), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(4, 4), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(4, 4), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=256, kernel_size=(4, 4), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(n_filter_out, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    if print_summary:
        model.summary()

    return model


if __name__ == "__main__":

    if os.path.exists(pickle_file):
        with open(pickle_file, "rb") as f:
            X, y, name_to_idx, idx_to_name = pickle.load(f)
    else:
        exit(pickle_file + " does not exist")

    print(name_to_idx)
    print(idx_to_name)
    # preprocess input
    X = X / 255.0  # X.max()

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    y_cat_train = to_categorical(y_train, 2)
    y_cat_test = to_categorical(y_test, 2)

    print(type(X_train), X_train.min(), X_train.max(), X_train.shape, X_train[0].dtype)
    print(type(y_cat_test), y_cat_test.min(), y_cat_test.max(), y_cat_test.shape, y_cat_test[0].dtype)

    input_shape = X_train[0].shape

    model = model_224(2, True)

    early_stop = EarlyStopping(monitor='val_loss', patience=5)

    model.fit(X_train, y_cat_train, epochs=50, validation_data=(X_test, y_cat_test), batch_size=128,
              callbacks=[early_stop])

    tstamp = time.strftime("%Y%m%d-%H%M%S")
    model.save(os.path.join(os.path.dirname(pickle_file), tstamp + '_model_224.h5'))

    losses = pd.DataFrame(model.history.history)

    # print(losses.head())

    with PdfPages(os.path.join(os.path.dirname(pickle_file), tstamp + '_val_accuracy.pdf')) as pdf:
        losses[['accuracy', 'val_accuracy']].plot()
        pdf.savefig()
        plt.close()

    with PdfPages(os.path.join(os.path.dirname(pickle_file), tstamp + '_val_loss.pdf')) as pdf:
        losses[['loss', 'val_loss']].plot()
        pdf.savefig()
        plt.close()

    print(model.evaluate(X_test, y_cat_test, verbose=0))
