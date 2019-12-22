import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

import os
import numpy as np
import matplotlib.pyplot as plt

from stdLib import stdDisp as disp, stdUtil as util
from dlLib import dlDataManager, dlDisp, dlModels, dlIO

# Load parameters
params = dlIO.InputManager("CatDog")

# Load dataset
datasetManager = dlDataManager.DataLoader(params)
dataset = datasetManager.dataset


sample_training_images, _ = next(dataset.train_images)
disp.gridShow(sample_training_images[0:5], 20,20,1,5,None)

model = dlModels.CatDog(params)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit_generator(
    dataset.train_images,
    steps_per_epoch= params.total_train_images // params.batch_size,
    epochs= params.epochs,
    validation_data= dataset.validation_images,
    validation_steps= params.total_validation_images // params.batch_size
)



dlDisp.plot_training_trend(params.epochs, history)



