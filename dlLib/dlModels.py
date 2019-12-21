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