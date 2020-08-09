#!/usr/bin/env python
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
import time
import json
import os
from os.path import join, dirname, basename, isfile, exists
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

batch_size = 32
vgg16_top_layers_train = 1
max_epochs = 50

def cnn8(input_shape, n_filter_out):
    model = Sequential()

    model.add(Conv2D(filters=16, kernel_size=(4, 4), input_shape=input_shape, activation='relu'))  # (224, 224, 3)
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=32, kernel_size=(4, 4), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(4, 4), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(4, 4), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=256, kernel_size=(4, 4), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

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
    parser = argparse.ArgumentParser(description="Train obstacle avoidance data recorded from robot webcam.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("method", help="Model used, choose among CNN8, VGG16, MobileNetV2", type=str)
    parser.add_argument("data_path",
                        help="Path to pickle file with data: X, y, name_to_idx, idx_to_name, or path to directory with train and test subdirs",
                        type=str)
    parser.add_argument("--val_rat", help="Ratio of the validation set", type=float, default=0.3)

    args = parser.parse_args()

    # create output directory
    out_dir = join(dirname(args.data_path),
                   basename(args.data_path) + "_" + args.method + "_" + time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir)

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
            train_path = join(args.data_path, "train")
            test_path = join(args.data_path, "test")
            img_shape = read_shape_first_train_image(args.data_path)

            image_gen = ImageDataGenerator(rotation_range=20,  # rotate the image 20 degrees
                                           width_shift_range=0.10,  # Shift the pic width by a max of 5%
                                           height_shift_range=0.10,  # Shift the pic height by a max of 5%
                                           rescale=1. / 255,  # Rescale the image by normalzing it.
                                           shear_range=0.1,  # Shear means cutting away part of the image (max 10%)
                                           zoom_range=0.1,  # Zoom in by 10% max
                                           horizontal_flip=True,  # Allo horizontal flipping
                                           fill_mode='nearest')  # Fill in missing pixels with the nearest filled value

            os.makedirs(os.path.join(out_dir, "augs"))

            train_image_gen = image_gen.flow_from_directory(train_path,
                                                            target_size=img_shape[:2],
                                                            color_mode='rgb',
                                                            batch_size=batch_size,
                                                            class_mode='binary',
                                                            save_to_dir=os.path.join(out_dir, "augs"),
                                                            save_prefix='train',
                                                            save_format='jpg')

            test_image_gen = image_gen.flow_from_directory(test_path,
                                                           target_size=img_shape[:2],
                                                           color_mode='rgb',
                                                           batch_size=batch_size,
                                                           class_mode='binary',
                                                           save_to_dir=os.path.join(out_dir, "augs"),
                                                           save_prefix='val',
                                                           save_format='jpg',
                                                           shuffle=False)

            print(train_image_gen.class_indices, test_image_gen.class_indices)
    else:
        exit("Path " + args.data_path + " does not exist")

    # compile method based on selected model
    model = None

    if args.method == "CNN8":
        model = cnn8(img_shape, 2 if isfile(args.data_path) else 1)
    elif args.method == "VGG16":
        model = vgg16plus(img_shape, 2 if isfile(args.data_path) else 1, vgg16_top_layers_train)
        # print(model.summary())
        # exit("DEBUG")
    else:
        exit("Method " + args.method + " cound not be found")

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=5)

    if isfile(args.data_path):
        model.fit(X_train, y_cat_train, validation_data=(X_test, y_cat_test), epochs=max_epochs, batch_size=batch_size, callbacks=[early_stop])
    else:
        model.fit(train_image_gen, validation_data=test_image_gen, epochs=max_epochs, batch_size=batch_size) # , callbacks=[early_stop]

    # print(losses.head())
    # print(model.evaluate(X_test, y_cat_test, verbose=0))

    if model is not None:
        model.save(join(out_dir, args.method + '.h5'))

        f = open(join(out_dir, 'idx_to_name.json'), "w")
        f.write(json.dumps(idx_to_name))
        f.close()

        f = open(join(out_dir, 'name_to_idx.json'), "w")
        f.write(json.dumps(name_to_idx))
        f.close()

        losses = pd.DataFrame(model.history.history)

        with PdfPages(join(out_dir, args.method + '_val_acc.pdf')) as pdf:
            losses[['accuracy', 'val_accuracy']].plot()
            plt.tight_layout()
            plt.grid(True)
            pdf.savefig()
            plt.close()

        with PdfPages(join(out_dir, args.method + '_val_loss.pdf')) as pdf:
            losses[['loss', 'val_loss']].plot()
            plt.tight_layout()
            plt.grid(True)
            pdf.savefig()
            plt.close()
