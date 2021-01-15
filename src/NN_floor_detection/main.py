"""
Main
"""

from core.nn_model import build_model
from core.dataset import generate_dataset, generate_test_dataset

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
# path to train folder
test_folder_path = Path("data/Dataset_ready/")
# path for saving model
model_path = ".data/model_save"
# input image size
im_size = (100, 100)
# path for saving checkpoints
checkpoint_path = './results/training_checkpoints/weights.{epoch:02d}.ckpt'
# labels for clases - SVHN dataset
CLASS_NAMES_SVHN = [
        'Num: 1', 'Num: 2', 'Num: 3',
        'Num: 4', 'Num: 5', 'Num: 6',
        'Num: 7', 'Num: 8', 'Num: 9', 'Num: 0']

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


def predict_image(model, PATH_TO_IMG, image_size):
    img = keras.preprocessing.image.load_img(
        PATH_TO_IMG,
        target_size=image_size,
        color_mode="rgb",
        grayscale=False
    )

    img_array = keras.preprocessing.image.img_to_array(img)

    tf_img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(tf_img_array)
    result = CLASS_NAMES_SVHN[predictions.argmax(axis=1)[0]]

    print(
        "Predictions: ", predictions
    )

    plt.imshow(img_array/255.0)
    plt.title(result)
    plt.show()

    print("Labels: ", result)


def predict_img_loop(model, path_read, image_size=(32, 32)):
    # go trough all folders in the path
    for fo_number, folder in enumerate(os.listdir(path_read)):
        print("new folder")
        for fi_number, filename in enumerate(os.listdir(path_read+'\\'+folder)):
            if fi_number > 20:
                break
            else:
                img_path = path_read+str(folder)+'\\'+str(filename)
                img = keras.preprocessing.image.load_img(
                    img_path,
                    target_size=image_size,
                    color_mode="rgb",
                    grayscale=False
                )

                img_array = keras.preprocessing.image.img_to_array(img)

                tf_img_array = tf.expand_dims(img_array, 0)  # Create batch axis

                predictions = model.predict(tf_img_array)
                result = CLASS_NAMES_SVHN[predictions.argmax(axis=1)[0]]

                print(
                    "Predictions: ", predictions
                )

                plt.imshow(img_array/255.0)
                plt.title(result)
                plt.show()

                print("Labels: ", result)


def predict_dataset(model, IMG_SIZE):
    test_ds = generate_test_dataset(test_folder_path, image_size=IMG_SIZE)
    print("Evaluate")
    result = model.evaluate(test_ds)

# TODO: create function for training with saved weights


if __name__ == "__main__":
    """
    variables
    """
    NUM_CLASSES = 3
    IMG_SIZE = (32, 32)
    PATH_TO_IMG = Path("data/test/img_test/6_frame_014762.jpeg")
    PATH_TO_WEIGHS = Path("results/training_checkpoints/")
    PATH_TO_MODEL = "data/models/svhn_best_cnn.h5"
    path_read = ('data/Dataset_ready/')

    # train = True --> train neural network
    # train = False --> load model or weights and eveluate network
    TRAIN = False

    # LOAD_MODEL = True --> load model
    # LOAD_MODEL = False --> load weights
    LOAD_MODEL = True

    if TRAIN is True:
        train(NUM_CLASSES)
    else:
        if LOAD_MODEL is True:
            model = keras.models.load_model(PATH_TO_MODEL)
            print(model)
        else:
            model_old = build_model(NUM_CLASSES)
            model = load_w(model_old, PATH_TO_WEIGHS)

        # predict_image(model, PATH_TO_IMG, IMG_SIZE)
        # predict_dataset(model, IMG_SIZE)
        predict_img_loop(model, path_read)
