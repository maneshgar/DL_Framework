import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from stdLib import stdDisp as disp, stdUtil as util
from dlLib import dlDataManager, dlDisp, dlModels, dlIO

print(tf.__version__)

params = dlIO.InputManager("FashionMnist")
datasetManager = dlDataManager.DataLoader(params)
dataset = datasetManager.dataset

model = dlModels.dense(params.source_shape, params.num_classes)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(dataset.train_images, dataset.train_labels, epochs=params.epochs)

test_loss, test_acc = model.evaluate(dataset.validation_images,  dataset.validation_labels, verbose=2)

print('\nTest accuracy:', test_acc)

validation_predictions = model.predict(dataset.validation_images)

dlDisp.plot_predictions(dataset.validation_images, dataset.validation_labels, validation_predictions, params.class_names)