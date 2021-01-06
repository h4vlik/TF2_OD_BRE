import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt
from pathlib import Path


def build_model(NUM_CLASSES):
    model = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        pooling="avg"
    )

    model.trainable = False

    # custom modifications on top of pre-trained model and fit
    model_new = tf.keras.models.Sequential()
    model_new.add(model)
    model_new.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))

    model_new.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    return model_new


if __name__ == "__main__":
    pass
