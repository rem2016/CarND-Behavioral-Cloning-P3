# -*- coding: utf-8 -*-
"""

@author: Rem
@contack: remch183@outlook.com
@time: 2017/05/24/ 10:29 
"""
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import cv2
from random import shuffle
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
import csv

samples = []
with open('./mydata/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# Reading data
def generator(samples, batch_size=32):
    num_samples = len(samples)
    batch_size //= 2
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './mydata/IMG/'+batch_sample[0].split('\\')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                flip_image = np.fliplr(center_image)
                flip_angle = -center_angle
                images.append(flip_image)
                angles.append(flip_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator, validation_generator = \
    generator(train_samples), generator(validation_samples)


# Model
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Flatten())
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit_generator(
    train_generator,
    samples_per_epoch=len(train_samples),
    validation_data=validation_generator,
    nb_val_samples=len(validation_samples),
    nb_epoch=5
)
model.save('model.h5')




