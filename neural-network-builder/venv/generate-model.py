import os
import shutil
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


def load_cifar10_data(data_dir):
    train_images = []
    train_labels = []

    for i in range(1, 6):
        with open(os.path.join(data_dir, f'data_batch_{i}'), 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
            images = data_dict[b'data']
            labels = data_dict[b'labels']

            train_images.extend(images)
            train_labels.extend(labels)

    train_images = np.array(train_images).reshape(-1, 3,
                                                  32, 32).transpose(0, 2, 3, 1)
    train_labels = np.array(train_labels)

    with open(os.path.join(data_dir, 'test_batch'), 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
        test_images = data_dict[b'data'].reshape(
            -1, 3, 32, 32).transpose(0, 2, 3, 1)
        test_labels = np.array(data_dict[b'labels'])

    return (train_images, train_labels), (test_images, test_labels)


def build_model():
    # Define the model architecture
    model = models.Sequential()
    model.add(layers.Conv2D(
        32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Add dense layers on top
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    return model


def train_model(model, train_images, train_labels, test_images, test_labels):
    # Compile and train the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=50,
                        validation_data=(test_images, test_labels))

    # Check if the model directory exists
    if os.path.exists('model'):
        # If it does, delete it
        shutil.rmtree('model')
    # Recreate the model directory
    os.makedirs('model')

    # Save the model
    model.save('model/cifar-10-batches-py-model.keras')

    return history


# Load and preprocess the CIFAR10 dataset
data_dir = 'cifar-10-batches-py'
(train_images, train_labels), (test_images,
                               test_labels) = load_cifar10_data(data_dir)

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Build and train the model
model = build_model()
history = train_model(model, train_images, train_labels,
                      test_images, test_labels)

# Print the history dictionary
print(history.history)
