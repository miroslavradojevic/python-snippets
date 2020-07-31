#!/usr/bin/env python
import os
import time
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib.image import imread
from train_cnn import model_224
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

data_dir = "/media/miro/WD/jetbot_obstacle_avoidance/data_224_pckl_0.4"
print(os.listdir(data_dir))
train_path = os.path.join(data_dir, "train")  # "/media/miro/WD/jetbot_obstacle_avoidance/data_224_pckl_0.4/train"
test_path = os.path.join(data_dir, "test")  # "/media/miro/WD/jetbot_obstacle_avoidance/data_224_pckl_0.4/test"

train_class_path = os.path.join(train_path, os.listdir(train_path)[0])
image_path = os.path.join(train_class_path, os.listdir(train_class_path)[0])

print(image_path)
img = imread(image_path)
print(img.shape)  # (224, 224, 3)

image_gen = ImageDataGenerator(rotation_range=20,  # rotate the image 20 degrees
                               width_shift_range=0.10,  # Shift the pic width by a max of 5%
                               height_shift_range=0.10,  # Shift the pic height by a max of 5%
                               rescale=1 / 255,  # Rescale the image by normalzing it.
                               shear_range=0.1,  # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1,  # Zoom in by 10% max
                               horizontal_flip=True,  # Allo horizontal flipping
                               fill_mode='nearest'  # Fill in missing pixels with the nearest filled value
                               )

plt.imshow(img)
# plt.show()

plt.imshow(image_gen.random_transform(img))
# plt.show()

model = model_224(1)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 64

train_image_gen = image_gen.flow_from_directory(train_path,
                                                target_size=img.shape[:2],
                                                color_mode='rgb',
                                                batch_size=batch_size,
                                                class_mode='binary')

test_image_gen = image_gen.flow_from_directory(test_path,
                                               target_size=img.shape[:2],
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary', shuffle=False)

print(train_image_gen.class_indices)
print(test_image_gen.class_indices)

early_stop = EarlyStopping(monitor='val_loss', patience=5)
results = model.fit_generator(train_image_gen, epochs=100, validation_data=test_image_gen, callbacks=[early_stop])

tstamp = time.strftime("%Y%m%d-%H%M%S")
model.save(os.path.join(os.path.dirname(data_dir), tstamp + '_model_224_aug.h5'))

losses = pd.DataFrame(model.history.history)

with PdfPages(os.path.join(os.path.dirname(data_dir), tstamp + '_val_accuracy.pdf')) as pdf:
    losses[['accuracy', 'val_accuracy']].plot()
    pdf.savefig()
    plt.close()

with PdfPages(os.path.join(os.path.dirname(data_dir), tstamp + '_val_loss.pdf')) as pdf:
    losses[['loss', 'val_loss']].plot()
    pdf.savefig()
    plt.close()
