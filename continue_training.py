# -*- coding: utf-8 -*-
"""

@author: Rem
@contack: remch183@outlook.com
@time: 2017/05/27/ 17:02 
"""


from keras.models import load_model
import numpy as np
import random
import sklearn
import os
from sklearn.model_selection import train_test_split
import cv2
from random import shuffle
import pickle
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Nadam
from keras.callbacks import ModelCheckpoint
import csv


# data_root_path = './mydata/'
# data_root_path = '../Behavior Cloning/data/'
data_root_path = './data/'
if not os.path.exists(data_root_path):
    data_root_path = '../Behavior Cloning/data/'

with open('pickle/train.p', 'rb') as f:
    train_samples = pickle.load(f)
with open('pickle/valid.p', 'rb') as f:
    validation_samples = pickle.load(f)


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
                type_id = batch_sample[-1]
                end = batch_sample[type_id].split('\\')[-1]
                if '/' in end:
                    end = end.split('/')[-1]
                if 'jpg' not in end:
                    continue
                name = data_root_path + 'IMG/' + end
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3]) + shifts[type_id]
                if random.random() > 0.5:
                    images.append(center_image)
                    angles.append(center_angle)
                else:
                    flip_image = np.fliplr(center_image)
                    flip_angle = -center_angle
                    images.append(flip_image)
                    angles.append(flip_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator, validation_generator = \
    generator(train_samples), generator(validation_samples, argue=False)


model = load_model("./model_shuffle_typeid.h5")

