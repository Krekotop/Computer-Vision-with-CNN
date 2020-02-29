# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 20:09:27 2020

@author: fredh
"""
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers.core import Dropout
from keras.layers import Flatten
from keras.layers import Dense
import numpy as np
from keras.preprocessing import image

classifier = Sequential()
classifier.add(Convolution2D(30,(2,2), input_shape=(218,218,3), activation = 'relu')) 
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Convolution2D(66,(2,2), input_shape=(218,218,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2))) 
classifier.add(Flatten())
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dropout(0.15))
classifier.add(Dense(units = 4, activation = 'softmax'))


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range = 40,
        width_shift_range=0.2,
        height_shift_range = 0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode = 'nearest',
        ) 

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(218,218),
        batch_size=20,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(218,218),
        batch_size=20,
        class_mode='categorical')
print(classifier.summary())
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier.fit_generator(
        training_set,
        workers=15,
        use_multiprocessing=False,
        steps_per_epoch=42,
        epochs=25,
        validation_data=test_set,
        validation_steps=17)

classifier.save_weights('ftp2.h5')
 
def ultimate_predictor(link):
    test_image = image.load_img(link, target_size = (218,218))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    training_set.class_indices
    if result[0][0] == 1:
        prediction = 'shaver'
    if result[0][1] == 1:
        prediction = 'smart-baby-bottle'
    if result[0][2] == 1:
        prediction = 'toothbrush'
    if result[0][3] == 1:
        prediction = 'wake-up-light'
    return prediction
     

