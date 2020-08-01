#!/usr/bin/env python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train.shape
x_train[0].shape
print(type(x_train), x_train.min(), x_train.max(), x_train.shape, x_train[0].dtype)
print(type(y_train), y_train.min(), y_train.max(), y_train.shape, y_train[0].dtype)

plt.imshow(x_train[0])
plt.imshow(x_train[12])

# PreProcessing
x_train = x_train/255
x_test = x_test/255

y_cat_train = to_categorical(y_train,10)
y_cat_test = to_categorical(y_test,10)

print(type(x_train), x_train.min(), x_train.max(), x_train.shape, x_train[0].dtype)
print(type(y_cat_train), y_cat_train.min(), y_cat_train.max(), y_cat_train.shape, y_cat_train[0].dtype)

if True:
    exit()

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(32, 32, 3), activation='relu',))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(32, 32, 3), activation='relu',))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss',patience=3)
model.fit(x_train,y_cat_train,epochs=15,validation_data=(x_test,y_cat_test),callbacks=[early_stop])
# Careful, don't overwrite our file!
# model.save('cifar_10epochs.h5')
losses = pd.DataFrame(model.history.history)
losses.head()
losses[['accuracy','val_accuracy']].plot()
losses[['loss','val_loss']].plot()
model.metrics_names
print(model.metrics_names)
print(model.evaluate(x_test,y_cat_test,verbose=0))
from sklearn.metrics import classification_report,confusion_matrix
predictions = model.predict_classes(x_test)
print(classification_report(y_test,predictions))
confusion_matrix(y_test,predictions)

plt.figure(figsize=(10,6))
sns.heatmap(confusion_matrix(y_test,predictions),annot=True)
# https://github.com/matplotlib/matplotlib/issues/14751

# Predicting a given image
my_image = x_test[16]
plt.imshow(my_image)
# SHAPE --> (num_images,width,height,color_channels)
model.predict_classes(my_image.reshape(1,32,32,3))
# 5 is DOG
# https://www.cs.toronto.edu/~kriz/cifar.html

