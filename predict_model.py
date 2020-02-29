# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 12:53:56 2020

@author: fredh
"""
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten
from keras.layers.core import Dropout
import os
import numpy as np
from keras.preprocessing import image

import matplotlib.pyplot as plt

classifier = Sequential()
classifier.add(Convolution2D(30,(2,2), input_shape=(218,218,3), activation = 'relu')) 
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Convolution2D(66,(2,2), input_shape=(218,218,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2))) 
classifier.add(Flatten())
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dropout(0.15))
classifier.add(Dense(units = 4, activation = 'softmax'))

def ultimate_predictor(img_dir, model):
    labels = []
    batch_holder = []
    for i,test_image in enumerate(os.listdir(img_dir)):
        test_image = image.load_img(os.path.join(img_dir,test_image), target_size=(218,218))
        batch_holder.append(test_image)
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)    
        if result[0][0] == 1:
            prediction = 'shaver'
        if result[0][1] == 1:
            prediction = 'smart-baby-bottle'
        if result[0][2] == 1:
            prediction = 'toothbrush'
        if result[0][3] == 1:
            prediction = 'wake-up-light'
        labels.append(prediction)
        fig = plt.figure(figsize=(20, 20))
    for i,img in enumerate(batch_holder):
        fig.add_subplot(4,5, i+1)
        plt.title(labels[i])
        plt.imshow(img)
    return labels   
classifier.load_weights("model_weights.h5")
img_dir='contest'

final_result = ultimate_predictor(img_dir,classifier)