import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt
from pathlib import Path


def build_my_model(num_classes, input_shape):
    """
    build_my_model
    - build model from scratch
    - compile

    :param NUM_CLASSES: number of classes to be trained

    :return model: model_my, keras model variable
    """
