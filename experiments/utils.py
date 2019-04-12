'''
Functions that are used by some experiments in this package.
'''
import keras
import tensorflow as tf
import numpy as np
from layers.quantized_layers import QuantizedDense
from layers.quantized_ops import quantized_relu

def get_scaled_fashion_mnist():
    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255.0

    test_images = test_images / 255.0
    return (train_images, train_labels), (test_images, test_labels)


def get_vanilla_NN():
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
    '''
    Model has same architecture as model returned from `get_vanilla_CNN`.
    :param q: number of bits for activation functions and weights
    '''
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        QuantizedDense(units=128, nb=q, activation=quantized_relu),
        QuantizedDense(units=10, nb=q, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def _get_positive_samples(images, labels, model, same_label):
    '''
    :param images: samples to classify
    :param labels: ground truth labels
    :param model: model that performs classification
    :param same_label: If true, return correctly classified images. If false, return not correctly classified images
    '''
    positive_images = []
    positive_labels = []
    for image, label in zip(images, labels):
        image = (np.expand_dims(image, 0))
        predicted_label = np.argmax(model.predict(image))
        if same_label:
            # those that are correctly classified
            if predicted_label == label:
                positive_images.append(image[0, :, :])
                positive_labels.append(label)
        else:
            # those that are not correctly classified
            if predicted_label != label:
                positive_images.append(image[0, :, :])
                positive_labels.append(label)

    np_images = np.array(positive_images)
    np_labels = np.array(positive_labels)
    return np_images, np_labels


def _filter_samples(images: np.ndarray, labels: np.ndarray, models: list, correctly_classified: bool):
    if len(models) == 1:
        return _get_positive_samples(images, labels, models[0], correctly_classified)

    positive_images, positive_labels = _get_positive_samples(images, labels, models[0], correctly_classified)

    return _filter_samples(positive_images, positive_labels, models[1:], correctly_classified)


def filter_correctly_classified_samples(images: np.ndarray, labels: np.ndarray, models: list):
    '''
    Return images and associated labels that are correctly classified by all models.
    :param models: list of models
    '''
    assert len(models) > 0, "List of models must not be empty"
    return _filter_samples(images, labels, models, True)


def filter_not_correctly_classifed_samples(images: np.ndarray, labels: np.ndarray, models: list):
    '''
    Return images and associated labels that are not correctly classified by all models.
    :param models: list of models
    '''
    assert len(models) > 0, "List of models must not be empty"
    return _filter_samples(images, labels, models, False)
