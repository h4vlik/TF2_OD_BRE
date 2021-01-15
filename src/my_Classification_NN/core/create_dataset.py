# create dataset
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from pathlib import Path


def generate_dataset(train_folder_path, image_size=(100, 100), batch_size=32):
    """
    generate dataset for training and also for validation
    """
    data_gen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.15,
        rotation_range=8,
        zoom_range=[0.95, 1.05],
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


def generate_test_dataset(dataset_folder_path, image_size=(100, 100), batch_size=32):
    """
    generate dataset for testing of model // not validation
    """
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_ds = test_datagen.flow_from_directory(
        dataset_folder_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="sparse")

    return test_ds


def imageVisualization(train_ds):
    plt.figure(figsize=(2, 2))
    images, labels = train_ds.next()
    for i in range(2):
        ax = plt.subplot(2, 1, i + 1)
        plt.imshow(images[i])
        plt.title(str(labels[i]))
        plt.axis("off")
    plt.show()


if __name__ == "__main__":
    dataset_folder_path = Path("data/Dataset_ready/")
    test_ds = generate_test_dataset(dataset_folder_path)
    imageVisualization(test_ds)
