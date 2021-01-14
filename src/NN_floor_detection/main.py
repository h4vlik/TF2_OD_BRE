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


def train(NUM_CLASSES, EPOCHS=10):
    train_folder_path = Path("data/Dataset_try/")
    model_path = Path("data/model_save/")
    # path for saving checkpoints
    checkpoint_path = './results/training_checkpoints/weights.{epoch:02d}.ckpt'

    train_ds, val_ds = generate_dataset(train_folder_path)

    model = build_model(NUM_CLASSES)

    checkpoint_callbacks = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        save_freq='epoch'
    )

    model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=[checkpoint_callbacks]
    )


def load(model, PATH_TO_WEIGHS):
    # The model weights (that are considered the best)
    # are loaded into the model.
    latest = tf.train.latest_checkpoint(PATH_TO_WEIGHS)
    model.load_weights(latest)

    return model

# TODO: create function for training with saved weights


if __name__ == "__main__":
    NUM_CLASSES = 3
    PATH_TO_WEIGHS = Path("results/training_checkpoints/")
    # train = True --> train neural network
    # train = False --> load save weights and eveluate network
    TRAIN = True

    if TRAIN is True:
        train(NUM_CLASSES)
    else:
        model_old = build_model(NUM_CLASSES)
        model = load(model_old, PATH_TO_WEIGHS)
        print(model)

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
