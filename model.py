# -*- coding: utf-8 -*-
"""

@author: Rem
@contack: remch183@outlook.com
@time: 2017/05/24/ 10:29 
"""
import numpy as np
import pickle
import random
import sklearn
import os
from sklearn.model_selection import train_test_split
import cv2
from random import shuffle
import keras as K
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import csv
random.seed(123)


# data_root_path = './mydata/'
# data_root_path = '../Behavior Cloning/data/'
data_root_path = './data/'
if not os.path.exists(data_root_path):
    data_root_path = '../Behavior Cloning/data/'

samples = []
with open(data_root_path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for index, line in enumerate(reader):
        if index == 0:
            continue
        steer = float(line[3])
        if abs(steer) < 0.05 and steer != 0.0:
            continue
        if steer == 0.0:
            if random.random() > 0.3:
                continue
        for type_id in range(3):
            samples.append(line + [type_id])


train_samples, validation_samples = train_test_split(samples, test_size=0.2, random_state=123)


# Reading data
def generator(samples, batch_size=300, argue=True):
    num_samples = len(samples)
    while 1:
        # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            shifts = [0., 0.07, -0.07]
            for batch_sample in batch_samples:
                # the type_id indicates where the image comes from.
                # center, left or right
                type_id = batch_sample[-1]
                # chose which image to use by type_id
                end = batch_sample[type_id].split('\\')[-1]
                if '/' in end:
                    end = end.split('/')[-1]
                if 'jpg' not in end:
                    continue
                name = data_root_path + 'IMG/' + end
                center_image = cv2.imread(name)
                # batch_sample[3] is the steering data
                center_angle = float(batch_sample[3])
                modified_angle = center_angle + shifts[type_id]

                # argumenting the input data
                if argue:
                    # Change the brightness and contract of images
                    k = random.random() * 0.6 + 0.7
                    B = random.random() * 0.8 - 0.4
                    center_image = (center_image - 127.5 * (1 - B)) * k + 127.5 * (1 + B)
                    center_image = center_image.astype(np.uint8)

                # randomly chose whether flip the data
                if random.random() > 0.5:
                    images.append(center_image)
                    angles.append(modified_angle)
                else:
                    flip_image = np.fliplr(center_image)
                    flip_angle = -modified_angle
                    images.append(flip_image)
                    angles.append(flip_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator, validation_generator = \
    generator(train_samples), generator(validation_samples, 300, argue=False)

with open('pickle/train.p', 'wb') as f:
    pickle.dump(train_samples, f)
with open('pickle/valid.p', 'wb') as f:
    pickle.dump(validation_samples, f)


# Model
model = Sequential()
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 128.0 - 1.0))
model.add(Convolution2D(24, 5, 5, activation='relu', border_mode='same', input_shape=(50, 320, 1)))
model.add(MaxPooling2D())
model.add(Convolution2D(36, 5, 5, activation='relu', border_mode='same'))
model.add(MaxPooling2D())
model.add(Convolution2D(48, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))


optimizer = Adam()
model.compile(loss='mse', optimizer=optimizer)
print(model.summary())
tensorboard_callback = K.callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=0,
    write_graph=True,
    write_images=True,
)
model.fit_generator(
    train_generator,
    samples_per_epoch=len(train_samples),
    validation_data=validation_generator,
    nb_val_samples=len(validation_samples),
    nb_epoch=90,
    verbose=1,
    callbacks=[
        ModelCheckpoint('model_619_elu.h5'),
        tensorboard_callback
    ]
)
model.save('model_619_elu.h5')

