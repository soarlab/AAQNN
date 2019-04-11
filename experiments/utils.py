'''
Functions that are used by some experiments in this package.
'''
import keras
import tensorflow as tf
import numpy as np


def get_scaled_fashion_mnist():
    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255.0

    test_images = test_images / 255.0
    return (train_images, train_labels), (test_images, test_labels)


def get_vanilla_CNN():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def get_QNN(q: int):
    return None


def _get_positive_samples(images, labels, model):
    '''
    Return images and associated labels that are correctly classified by given model.
    '''
    positive_images = []
    positive_labels = []
    for image, label in zip(images, labels):
        image = (np.expand_dims(image, 0))
        predicted_label = np.argmax(model.predict(image))
        if predicted_label == label:
            positive_images.append(image[0, :, :])
            positive_labels.append(label)

    np_images = np.array(positive_images)
    np_labels = np.array(positive_labels)
    return np_images, np_labels


def filter_positive_samples(images, labels, models):
    '''
    Return images and associated labels that are correctly classified by all models.
    :param models: list of models
    '''
    # there is no support for >1 model at the moment
    assert len(models) == 1
    return _get_positive_samples(images, labels, models[0])
