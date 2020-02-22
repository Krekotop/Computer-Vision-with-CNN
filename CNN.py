# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 20:09:27 2020

@author: fredh
"""
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()

#Convolution

classifier.add(Convolution2D(32,(3,3), input_shape=(64,64,3), activation = 'relu')) 
#if not proper results, try to increase input shape to 256 256

#Pooling - reducing size of feature maps
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Flatten())

classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 4, activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range = 40,
        width_shift_range=0.2,
        height_shift_range = 0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode = 'nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

classifier.fit_generator(
        training_set,
        workers=1,
        use_multiprocessing=False,
        steps_per_epoch=48,
        epochs=15,
        validation_data=test_set,
        validation_steps=16)