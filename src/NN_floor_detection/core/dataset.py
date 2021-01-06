# create dataset
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt
from pathlib import Path


def generate_dataset(train_folder_path, image_size=(224, 224), batch_size=32):
    data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,
        validation_split=0.2  # split data
    )

    train_ds = data_gen.flow_from_directory(
        train_folder_path,
        color_mode="rgb",
        target_size=image_size,
        batch_size=batch_size,
        class_mode="sparse",
        subset="training"
    )

    val_ds = data_gen.flow_from_directory(
        train_folder_path,
        color_mode="rgb",
        target_size=image_size,
        batch_size=batch_size,
        class_mode="sparse",
        subset="validation"
    )

    return train_ds, val_ds


def imageVisualization(train_ds, val_ds):
    plt.figure(figsize=(2, 2))
    images, labels = train_ds.next()
    for i in range(2):
        ax = plt.subplot(2, 1, i + 1)
        plt.imshow(images[i])
        plt.title(str(labels[i]))
        plt.axis("off")
    plt.show()


if __name__ == "__main__":
    train_folder_path = Path("data/Dataset_try/")
    train, val = generate_dataset(train_folder_path)
    imageVisualization(train, val)
