import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from stdLib import stdDisp as disp, stdUtil as util
from dlLib import dlDataManager, dlDisp, dlModels, dlIO

print(tf.__version__)

params = dlIO.InputManager("Name")
datasetManager = dlDataManager.DataLoader(params.dataset, params)
dataset = datasetManager.dataset

model = dlModels.dense(params.source_shape, params.num_classes)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(dataset.train_images, dataset.train_labels, epochs=10)

test_loss, test_acc = model.evaluate(dataset.test_images,  dataset.test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

test_predictions = model.predict(dataset.test_images)

dlDisp.plot_predictions(dataset.test_images, dataset.test_labels, test_predictions, params.class_names)