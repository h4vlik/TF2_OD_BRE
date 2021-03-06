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
import numpy as np
import datetime
from sklearn.metrics import classification_report, confusion_matrix
import cv2


"""
variables
"""
# not changable variables
# fix bug
ImageFile.LOAD_TRUNCATED_IMAGES = True
main_dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# path to train folder
train_folder_path = os.path.join(main_dir_path, r"data\\Dataset_ready\\")
# path for saving model
model_path = ".data/model_save"
# input image size
im_size = (224, 224)
# path for saving checkpoints
weights_checkpoint_path = os.path.join(
    main_dir_path,
    r'results\\training_checkpoints\\',
    os.path.basename("weights_new.{epoch:02d}"+str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))+".ckpt"))
# labels for clases - dataset
CLASS_NAMES = [
        'Num: 0', 'Num: 1', 'Num: 2', 'Num: 3',
        'Num: 4', 'Num: 5', 'Num: 6',
        'Num: 7', 'Num: 8', 'Num: 9']

tensorboard_efficient_name = "my_eficient_10class_224x224_{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

"""
functions
"""


def train(model, NUM_CLASSES, EPOCHS=50):
    train_ds, val_ds = generate_dataset(train_folder_path)

    # define tensorboard checkpoint
    log_dir = "logs/{}".format(tensorboard_efficient_name)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    checkpoint_callbacks = keras.callbacks.ModelCheckpoint(
        filepath=weights_checkpoint_path,
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
        callbacks=[checkpoint_callbacks, tensorboard_callback]
    )

    return model


def unfreeze_model(model):
    # We unfreeze the top 10 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-5:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
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
                result = CLASS_NAMES[predictions.argmax(axis=1)[0]]

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


def predict_confusion(model, img_size):
    """
    prediction on a validation dataset and compute confusion matrix
    """
    train_ds, val_ds = generate_dataset(train_folder_path)
    y_true = val_ds.classes  # numpy array of all labels
    print(y_true)
    y_pred = model.predict(val_ds, 32)
    y_pred = np.argmax(y_pred, axis=1)  # only biggest label
    print(y_pred)

    print('Confusion Matrix')
    print(confusion_matrix(y_true, y_pred))
    print('Classification Report')
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))


def predict_img_folder(model, PATH_TO_IMG, image_size):
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
        rounded_pred = np.around(predictions*100)
        result = CLASS_NAMES[predictions.argmax(axis=-1)[0]]

        print(
            "Predictions: ", rounded_pred
        )

        plt.imshow(img_array/255.0)
        plt.title(result)
        plt.show()

        print("Labels: ", result)


def camera_prediction(model, image_size):
    # define a video capture object
    vid = cv2.VideoCapture(0)

    while(True):
        # Capture the video frame
        # by frame
        ret, frame = vid.read()

        # Display the resulting frame
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    """
    variables
    """
    NUM_CLASSES = 10
    IMG_SIZE = (224, 224)
    PATH_TO_IMG = "data/test/img_test/"
    PATH_TO_WEIGHS = Path("results/training_checkpoints/colab_weigts_fine_tuning/")
    PATH_TO_MODEL = "data/models/svhn_best_cnn.h5"
    path_read = ('data/Dataset_ready/')

    # train = True --> train neural network
    # train = False --> load model or weights and eveluate network
    TRAIN = False
    CONTINUE = True

    # LOAD_MODEL = True --> load model
    # LOAD_MODEL = False --> load weights
    LOAD_MODEL = False

    if TRAIN is True:
        if CONTINUE is True:
            model_old = build_model(NUM_CLASSES)
            model = load_w(model_old, PATH_TO_WEIGHS)
            model = train(model, NUM_CLASSES)
        else:
            model = build_model(NUM_CLASSES)
            train(model, NUM_CLASSES)
    else:
        if LOAD_MODEL is True:
            model = keras.models.load_model(PATH_TO_MODEL)
            print(model)
        else:
            model_old = build_model(NUM_CLASSES)
            model = load_w(model_old, PATH_TO_WEIGHS)
            predict_confusion(model, IMG_SIZE)

        # predict_image(model, PATH_TO_IMG, IMG_SIZE)
        # predict_dataset(model, IMG_SIZE)
        # predict_img_folder(model, PATH_TO_IMG, image_size=IMG_SIZE)
