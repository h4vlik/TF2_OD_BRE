import matplotlib.pyplot as plt
from PIL import ImageFile
import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

"""
variables
"""
# not changable variables
# fix bug
ImageFile.LOAD_TRUNCATED_IMAGES = True
# main directory path
# main_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
main_dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# path to train folder
train_folder_path = os.path.join(main_dir_path, r"data\\Dataset_ready\\")

# path for loading weights checkpoints of efficient model
weights_checkpoint_path = os.path.join(
    main_dir_path,
    r'results\\training_checkpoints\\')

CLASS_NAMES_10 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def camera_save_img():
    # define a video capture object
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("test")

    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            img_name = "data/test/camera/cam_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()

    cv2.destroyAllWindows()


def camera_prediction(model):

    # define a video capture object
    cam = cv2.VideoCapture(0)

    # name of the window
    cv2.namedWindow("Display classification")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("ERROR - fail to grab the frame")
            break

        # show the image
        cv2.imshow("Display classification", frame)


if __name__ == "__main__":
    camera_save_img()
