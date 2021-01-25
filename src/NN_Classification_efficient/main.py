"""
Main
"""
import logging
from core.nn_model import build_model
from core.dataset import generate_dataset, generate_test_dataset

import matplotlib.pyplot as plt
from PIL import ImageFile, Image
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import datetime
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import time

tf.get_logger().setLevel(logging.ERROR)

"""
variables
"""
# not changable variables
# fix bug
ImageFile.LOAD_TRUNCATED_IMAGES = True
main_dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# path to train folder
train_folder_path = os.path.join(main_dir_path, r"data\\Dataset_ready\\")
# input image size
im_size = (224, 224)

# labels for clases - dataset
CLASS_NAMES = [
        'Num: 0', 'Num: 1', 'Num: 2', 'Num: 3',
        'Num: 4', 'Num: 5', 'Num: 6',
        'Num: 7', 'Num: 8', 'Num: 9']


"""
functions
"""


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


def predict_dataset(model, IMG_SIZE):
    test_ds = generate_test_dataset(test_folder_path, image_size=IMG_SIZE)
    print("Evaluate")
    result = model.evaluate(test_ds)


def predict_confusion(model, img_size):
    """
    prediction on a validation dataset and compute confusion matrix
    """
    # TODO: create confusion matrix code
    pass


def predict_img_folder(model, PATH_TO_IMG, image_size):
    for fi_number, filename in enumerate(os.listdir(PATH_TO_IMG)):
        start_time = time.time()  # define start time

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
        end_time = time.time()
        print("execution time is: ", (end_time - start_time))

        print(
            "Predictions: ", rounded_pred
        )

        plt.imshow(img_array/255.0)
        plt.title(result)
        plt.show()

        print("Labels: ", result)


def camera_prediction(model, image_size):
    # define a video capture object
    cam = cv2.VideoCapture(0)

    # name of the window
    cv2.namedWindow("Display classification")

    while True:
        start_time = time.time()  # define start time

        ret, frame = cam.read()

        # test image captured
        if not ret:
            print("ERROR - fail to grab the frame")
            break

        # if ESC pressed - exit
        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

        # show the image
        cv2.imshow("Display classification", frame)

        # convert cv2 image to PIL image
        # the color is converted from BGR to RGB
        color_coverted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # convert to PIL
        pil_frame = Image.fromarray(color_coverted)
        # convert to array
        frame_array = keras.preprocessing.image.img_to_array(pil_frame)

        frame_resized = keras.preprocessing.image.smart_resize(frame_array, (image_size))
        tf_img_array = tf.expand_dims(frame_resized, 0)  # Create batch axis

        # PREDICTION
        predictions = model.predict(tf_img_array)
        rounded_pred = np.around(predictions*100)
        result = CLASS_NAMES[predictions.argmax(axis=1)[0]]

        print(
            "Predictions: ", rounded_pred
        )
        print("\n Labels: ", result)

        end_time = time.time()  # end time
        print("execution time is: ", (end_time - start_time))


if __name__ == "__main__":
    """
    variables
    """
    NUM_CLASSES = 10
    IMG_SIZE = (224, 224)
    PATH_TO_WEIGHS = Path("results/training_checkpoints/colab_weigts_fine_tuning/")

    # main dataset path
    PATH_TO_DATASET = os.path.join(main_dir_path, r'data\\Dataset_ready\\')
    PATH_TO_IMG = os.path.join(main_dir_path, r'data\\test\\camera\\')

    """
    MAIN CODE
    """
    model_old = build_model(NUM_CLASSES)
    model = load_w(model_old, PATH_TO_WEIGHS)
    camera_prediction(model, IMG_SIZE)
    # predict_img_folder(model, PATH_TO_IMG, image_size=IMG_SIZE)
