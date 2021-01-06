"""
Main
"""

from core.nn_model import build_model
from core.dataset import generate_dataset

import matplotlib.pyplot as plt
from PIL import ImageFile

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt
from pathlib import Path

ImageFile.LOAD_TRUNCATED_IMAGES = True


def train(NUM_CLASSES, epochs=2):
    train_folder_path = Path("data/Dataset_try/")
    model_path = Path("data/model_save/")
    # checkpoint_path = Path("results/training_checkpoints/weights")
    checkpoint_path = '/results/weights'

    train_ds, val_ds = generate_dataset(train_folder_path)

    model = build_model(NUM_CLASSES)

    checkpoint_callbacks = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        save_weights_only=True,
        save_freq='epoch'
    )

    model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=[checkpoint_callbacks]
    )

    model.save_weights('/results/training_checkpoints/weights')


if __name__ == "__main__":
    NUM_CLASSES = 3

    train(NUM_CLASSES)

    """
    img = keras.preprocessing.image.load_img("XXXX", target_size=image_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    score = predictions[0]

    print(
        "This result for out image:"
        % (100 * (1 - score), 100 * score)
    )
    """
