# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

def dense(source_shape, num_classes):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=source_shape),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

###################################################################################
###################################################################################

def CatDog(params):
    model = keras.Sequential([
        keras.layers.Conv2D(16, 3, padding='same', activation='relu',
               input_shape=(params.source_shape[0], params.source_shape[1], params.source_channels)),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(params.num_classes, activation='sigmoid')
    ])
    return model