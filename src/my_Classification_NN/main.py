"""
Main script, HAVELKA, 15/01/2021
"""

from core.nn_my_model import build_my_model
from core.create_dataset import generate_dataset, generate_test_dataset

import matplotlib.pyplot as plt
from PIL import ImageFile
import datetime

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
train_folder_path = Path("data/Dataset_ready/")
# path for saving model checpoints
model_checkpoint_path = './results/model_save/my_cnn_model.h5'
# path for saving weights checkpoints
weights_checkpoint_path = './results/training_checkpoints/weights.{epoch:02d}.ckpt'
CLASS_NAMES = ['0', '1', '2']


def train(model, num_classes, image_shape, EPOCHS=50):
    """
    train
    - create dataset
    - define model
    - compile model
    - fit model (train model)

    :param num_classes: number of classes to be trained
    :param image_shape: format (100, 100, 3)
    :param EPOCHS: number of epochs

    :return trained_model: keras model variable
    """
    # create datasets
    train_ds, val_ds = generate_dataset(train_folder_path, image_size=image_shape[:-1])

    # define checkpoint - for saving best model
    checkpoint_callbacks = keras.callbacks.ModelCheckpoint(
        filepath=model_checkpoint_path,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        save_weights_only=False)

    # define tensorboard checkpoint
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # define compiler
    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"])

    # train it
    model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=[checkpoint_callbacks, tensorboard_callback]
    )

    return model


def transfer_train(num_classes, image_shape, path_to_model, path_to_save_model, EPOCHS=20, BATCH_SIZE=32):
    """
    :param num_classes: number of classes to be trained
    :param image_shape: format (100, 100, 3)
    :param path_to_model: path for loading pre-trained model
    :param path_to_save_model: path for saving model after transfer learning
    :param EPOCHS: number of epochs

    :return trained_model: keras model variable

    transfer_train
    - create dataset
    - load pre-trained model
    - freeze model
    - create new model on top
    - train top layer
    - finetuning - low lr, train entire model
    - fit model (train model)
    """
    # FIRST STEP

    train_ds, val_ds = generate_dataset(train_folder_path, image_size=image_shape[:-1], batch_size=BATCH_SIZE)

    model_pre = keras.models.load_model(path_to_model)  # load pre-trained model
    model_pre.pop()  # remove last layer
    model_pre.pop()  # remove last layer
    model_pre.pop()  # remove last layer

    model_pre.trainable = False  # freeze model

    inputs = keras.Input(shape=image_shape)  # define input
    x = model_pre(inputs, training=False)  # pre-trained model
    x = keras.layers.Dense(128, activation='relu')(x)  # new layer
    x = keras.layers.Dropout(0.4)(x)  # new layer
    outputs = keras.layers.Dense(num_classes)(x)  # new output layer
    model = keras.Model(inputs, outputs)  # create new model
    model.summary()

    # :TODO need probability output --> need sequential model - add sequential model obove
    # dont know how, but its necessary

    # define checkpoint - for saving best model
    checkpoint_callbacks = keras.callbacks.ModelCheckpoint(
        filepath=path_to_save_model,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        save_weights_only=False)

    # define tensorboard checkpoint
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # define compiler
    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"])

    # train it
    model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=[checkpoint_callbacks, tensorboard_callback]
    )

    # SECOND STEP - FINE TUNING - optional
    # :TODO fine tunning

    """
    model_new = keras.Sequential()  # create new model
    for layer in model_pre.layers[:-1]:  # go through until last layer
        model_new.add(layer)

    inputs = keras.Input(shape=image_shape)  # define input
    x = model_new(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(num_classes)(x)  # new output layer
    model = keras.Model(inputs, outputs)
    """
    return model


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


def predict_image(model, PATH_TO_IMG, image_size):
    for fi_number, filename in enumerate(os.listdir(PATH_TO_IMG)):
        img_path = PATH_TO_IMG+str(filename)
        img = keras.preprocessing.image.load_img(
            img_path,
            target_size=image_size,
            color_mode="rgb",
            grayscale=False
        )

        img_array = keras.preprocessing.image.img_to_array(img)

        tf_img_array = tf.expand_dims(img_array, 0)  # Create batch axis

        predictions = model.predict(tf_img_array)
        result = CLASS_NAMES[predictions.argmax(axis=1)[0]]

        print(
            "Predictions: ", predictions
        )

        plt.imshow(img_array/255.0)
        plt.title(result)
        plt.show()

        print("Labels: ", result)


if __name__ == "__main__":
    """
    variables
    """
    NUM_CLASSES = 10  # number of classes
    EPOCHS = 150  # epochs count
    BATCH_SIZE = 10  # batch size
    image_shape = (32, 32, 3)  # input image shape

    path_to_model_tfl = 'data/models/svhn_best_cnn.h5'  # saved model path, for loading saved models
    path_to_save_model_tfl = 'results/model_save/transfer/my_transfer_model_3.h5'

    path_to_trained_model = 'results/model_save/from_scratch/'  # saved model path, for loading saved models

    path_to_dataset = 'data/Dataset_ready/'  # main dataset path
    path_to_img = 'data/test/img_test/'
    TRAIN = True  # = True --> train neural network, = False --> load model or weights and eveluate network
    TRANSFER = True  # = True --> treansfer learning, use pre-trained model, = False --> train own model
    SCRATCH = False  # = True --> learn from 0, = False --> continue with training of saved model
    LOAD_MODEL = True  # = True --> load model, = False --> load weights

    if TRAIN is True:
        print("Training is starting ...")
        if TRANSFER is True:
            transfer_train(
                NUM_CLASSES,
                image_shape,
                path_to_model_tfl,
                path_to_save_model_tfl,
                EPOCHS=EPOCHS,
                BATCH_SIZE=BATCH_SIZE
            )
        else:
            if SCRATCH is True:
                # train from scratch
                model = build_my_model(image_shape, NUM_CLASSES)  # create model
                train(model, NUM_CLASSES, image_shape, EPOCHS=EPOCHS)  # train model
            else:
                # continue with training
                model = keras.models.load_model(path_to_save_model_tfl)
                train(model, NUM_CLASSES, image_shape, EPOCHS=EPOCHS)  # train model
    else:
        if LOAD_MODEL is True:
            print("Load model and start predictions on images ...")
            model = keras.models.load_model(path_to_save_model_tfl)
            predict_image(model, path_to_img, image_shape[:-1])
            print("Prediction is done.")
        else:
            model_old = build_my_model(image_shape, NUM_CLASSES)
            # TODO: use function for loading weights to model
