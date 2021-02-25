"""
Main
"""
from core.camera_estimation.nn_model import build_model
from main.flags_global import FLAGS
from input_feed import InputFeed

import matplotlib.pyplot as plt
from PIL import ImageFile, Image
import os
import pylab

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


class cameraEstimationFloor(object):
    def __init__(self):
        # :TODO IputFeed nemuze to takto fungovat Inicializuji 2x to same, takze by se spustil znovu akcelerometr :/
        self.main_dir = FLAGS.main_dir_path
        self.PATH_TO_WEIGHS = Path("results/training_checkpoints/colab_weigts_fine_tuning/")
        print(self.PATH_TO_WEIGHS)
        self.NUM_CLASSES = FLAGS.num_classes

        # labels for clases - dataset
        self.CLASS_NAMES = [
                '0.', '1.', '2.', '3.',
                '4.', '5.', '6.',
                '7.', '8.', '9.']
        self.IMG_SIZE = (224, 224)
        self.floor_estimation_camera = 0

        self.model = build_model(self.NUM_CLASSES)
        self.__load_weights()

    def __load_weights(self):
        # The model weights (that are considered the best)
        # are loaded into the model.
        latest = tf.train.latest_checkpoint(self.PATH_TO_WEIGHS)
        self.model.load_weights(latest)

    def camera_prediction(self, frame):
        # get image from video or camera in RGB
        # convert to PIL
        pil_frame = Image.fromarray(frame)
        # convert to array
        frame_array = keras.preprocessing.image.img_to_array(pil_frame)

        frame_resized = keras.preprocessing.image.smart_resize(frame_array, (self.IMG_SIZE))
        tf_img_array = tf.expand_dims(frame_resized, 0)  # Create batch axis

        # PREDICTION
        predictions = self.model.predict(tf_img_array)
        rounded_pred = np.around(predictions*100)
        result = self.CLASS_NAMES[predictions.argmax(axis=1)[0]]

        self.floor_estimation_camera = float(result)

    def spin(self, frame):
        """
        prediction on the picture from video or camera

        input: frame --> RGB image from cv2

        output: floor_estimation_camera
        """
        self.camera_prediction(frame)

        return np.copy(self.floor_estimation_camera)
