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


"""
variables
"""
# not changable variables
# fix bug
ImageFile.LOAD_TRUNCATED_IMAGES = True
# path to train folder
train_folder_path = Path("data/Dataset_try/")
# path for saving model
model_path = ".data/model_save"
# input image size
im_size = (100, 100)
# path for saving checkpoints
checkpoint_path = './results/training_checkpoints/weights.{epoch:02d}.ckpt'

"""
functions
"""


def train(NUM_CLASSES, EPOCHS=10):
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


def load_w(model, PATH_TO_WEIGHS):
    # The model weights (that are considered the best)
    # are loaded into the model.
    latest = tf.train.latest_checkpoint(PATH_TO_WEIGHS)
    model.load_weights(latest)

    return model


def predict_image(model, PATH_TO_IMG):
    img = keras.preprocessing.image.load_img(
        PATH_TO_IMG,
        target_size=im_size,
        color_mode="grayscale",
        grayscale=True
    )

    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    print(
        "Predictions: ", predictions
    )

    score = predictions[0]
    score0 = score[0]
    score1 = score[1]
    score2 = score[2]

    print(
        "This image is %.2f percent 0"
        % (100 * score0), '\n'
        "This image is %.2f percent 1"
        % (100 * score1), '\n'
        "This image is %.2f percent 2"
        % (100 * score2)
    )


# TODO: create function for training with saved weights


if __name__ == "__main__":
    """
    variables
    """
    NUM_CLASSES = 3
    PATH_TO_WEIGHS = Path("results/training_checkpoints/")
    PATH_TO_MODEL = "results/model_save"

    # train = True --> train neural network
    # train = False --> load model or weights and eveluate network
    TRAIN = True

    # LOAD_MODEL = True --> load model
    # LOAD_MODEL = False --> load weights
    LOAD_MODEL = False

    if TRAIN is True:
        train(NUM_CLASSES)
    else:
        if LOAD_MODEL is True:
            model = keras.models.load_model(PATH_TO_MODEL)
        else:
            model_old = build_model(NUM_CLASSES)
            model = load_w(model_old, PATH_TO_WEIGHS)
