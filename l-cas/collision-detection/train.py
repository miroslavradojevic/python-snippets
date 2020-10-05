#!/usr/bin/env python
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.image import imread
import time
import json
import os
from os.path import join, dirname, basename, isfile, isdir, exists
from sklearn.model_selection import train_test_split
from matplotlib.backends.backend_pdf import PdfPages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

# from tensorflow.keras.applications.resnet50 import ResNet50

batch_size = 128
vgg16_top_layers_train = 1
max_epochs = 100


def cnn8(input_shape, n_filter_out):
    print(input_shape, n_filter_out)
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=input_shape, padding='same',
                     activation='relu'))  # (224, 224, 3)
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    # model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
    # model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(n_filter_out, activation='sigmoid'))

    return model


def vgg16plus(input_shape, nr_filter_out, nr_trainable_top_layers):
    vgg = VGG16(include_top=False, weights="imagenet", input_shape=input_shape)  # pooling="avg"

    nr_layers_to_freeze = len(vgg.layers) - nr_trainable_top_layers

    for lay in vgg.layers[:nr_layers_to_freeze]:
        lay.trainable = False
    for lay in vgg.layers[nr_layers_to_freeze:]:
        lay.trainable = True

    vgg1 = Model(inputs=vgg.inputs, outputs=vgg.get_layer('block5_pool').output)

    model = Sequential()
    model.add(vgg1)

    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same'))
    # model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=1024, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    # model.add(GlobalAveragePooling2D())

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    # model.add(BatchNormalization())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nr_filter_out, activation='sigmoid'))

    # model.layers[0].trainable = False  # vgg1 is freezed

    return model


def mobilenetv2plus(input_shape, nr_filter_out, nr_trainable_top_layers):
    mobilenet = MobileNetV2(include_top=False, weights="imagenet", input_shape=input_shape)


def read_shape_first_train_image(data_path):
    train_path = join(data_path, "train")
    class_path = join(train_path, os.listdir(train_path)[0])
    image_path = join(class_path, os.listdir(class_path)[0])
    img = imread(image_path)
    return img.shape


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train collision detection",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-method", type=str, required=True,
                        help="Model, choose among CNN8, VGG16, MobileNetV2")
    parser.add_argument("-data_path", type=str, required=True,
                        help="Path to pickle file: X, y, name_to_idx, idx_to_name, or path to directory with train and test subdirs")
    parser.add_argument("-val_rat", type=float, default=0.3,
                        help="Ratio of the validation set")

    args = parser.parse_args()

    # create output directory
    outdir = join(Path(args.data_path).parent,
                  basename(args.data_path) + "_" + args.method + "_" + time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(outdir)

    if exists(args.data_path):
        if isfile(args.data_path):
            with open(args.data_path, "rb") as f:
                X, y, name_to_idx, idx_to_name = pickle.load(f)

            print(X.shape, y.shape, name_to_idx, idx_to_name)

            # preprocess input
            X = X / 255.0  # X.max()

            # train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.val_rat, random_state=42)

            y_cat_train = to_categorical(y_train, 2)
            y_cat_test = to_categorical(y_test, 2)

            print("X_train", type(X_train), X_train.min(), X_train.max(), X_train.shape, X_train[0].dtype)
            print("y_cat_train", type(y_cat_train), y_cat_train.min(), y_cat_train.max(), y_cat_train.shape,
                  y_cat_train[0].dtype)

            img_shape = X_train[0].shape

    else:
        exit("Path {} does not exist".format(args.data_path))

    # compile method based on selected model
    model = None

    if args.method == "CNN8":
        model = cnn8(img_shape, 2)
        model.summary()
    elif args.method == "VGG16":
        model = vgg16plus(img_shape, 2, vgg16_top_layers_train)
        # print(model.summary())
        # exit("DEBUG")
    else:
        exit("Method " + args.method + " cound not be found")

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=5)

    if isfile(args.data_path):
        model.fit(X_train, y_cat_train, validation_data=(X_test, y_cat_test), epochs=max_epochs,
                  batch_size=batch_size)  # , callbacks=[early_stop]

    # print(losses.head())
    # print(model.evaluate(X_test, y_cat_test, verbose=1))

    if model is not None:
        model.save(join(outdir, args.method + '.h5'))

        f = open(join(outdir, 'idx_to_name.json'), "w")
        f.write(json.dumps(idx_to_name))
        f.close()

        f = open(join(outdir, 'name_to_idx.json'), "w")
        f.write(json.dumps(name_to_idx))
        f.close()

        losses = pd.DataFrame(model.history.history)

        with PdfPages(join(outdir, args.method + '_val_acc.pdf')) as pdf:
            losses[['accuracy', 'val_accuracy']].plot()
            plt.tight_layout()
            plt.grid(True)
            pdf.savefig()
            plt.close()

        with PdfPages(join(outdir, args.method + '_val_loss.pdf')) as pdf:
            losses[['loss', 'val_loss']].plot()
            plt.tight_layout()
            plt.grid(True)
            pdf.savefig()
            plt.close()
