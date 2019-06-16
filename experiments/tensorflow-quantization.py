'''
This experiment uses quantization-aware training https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/quantize
provided by the tensorflow lite framework.

The idea is to achieve the same results using the tensorflow lite framework as with the framework used in the rest of this project.
If this turns out to be true, then the current quantization framework is sound. Otherwise, it is important to understand why the results are not the same.

Code base is built on top of https://colab.research.google.com/gist/ohtaman/c1cf119c463fd94b0da50feea320ba1e/edgetpu-with-keras.ipynb#scrollTo=3aluRnFKRN9y
'''


import tensorflow as tf
import keras
from cleverhans.attacks import ProjectedGradientDescent
from cleverhans.utils_keras import KerasModelWrapper
from experiments.utils import get_fashion_mnist, filter_correctly_classified_samples, get_stats
import numpy as np
import matplotlib.pyplot as plt


print(tf.__version__)

EPS = 0.06
FGSM_PARAMS = {'clip_min': 0.,
               'clip_max': 1.,
               'eps': EPS,
               # as in the original paper
               'nb_iter': int(min(EPS * 255 + 4, 1.25 * EPS * 255)),
               'rand_init': 0.
               }

# get dataset
(train_images, train_labels), (test_images, test_labels) = get_fashion_mnist()


def build_keras_model():
    return keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(units=128, activation=tf.nn.relu),
        keras.layers.Dense(units=10, activation=tf.nn.softmax)
    ])

# train 4 bits
train_graph = tf.Graph()
train_sess = tf.Session(graph=train_graph)

keras.backend.set_session(train_sess)
with train_graph.as_default():
    train_model = build_keras_model()

    # quantization aware training
    #tf.contrib.quantize.create_training_graph(input_graph=train_graph)
    tf.contrib.quantize.experimental_create_training_graph(input_graph=train_graph, weight_bits=4, activation_bits=4)

    train_sess.run(tf.global_variables_initializer())

    train_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    train_model.fit(train_images, train_labels, epochs=2)

    tf.contrib.quantize.experimental_create_eval_graph(input_graph=train_graph, weight_bits=4, activation_bits=4)

    _, test_acc = train_model.evaluate(test_images, test_labels, verbose=0)
    print("Test accuracy of QNN: " + str(test_acc))

    # perform attack on the QNN
    print("Generating adversarial samples for QNN..")
    wrap = KerasModelWrapper(train_model)
    iterative_fgsm = ProjectedGradientDescent(wrap, train_sess)
    adv_4_bits = iterative_fgsm.generate_np(test_images, **FGSM_PARAMS)
    print("Finished generating adversarial samples")

    # quantify perturbation
    mean, std, min, max = get_stats(np.array([np.linalg.norm(x - y) for x, y in zip(test_images, adv_4_bits)]))
    print("Information about L2 distances between adversarial and original samples:")
    print("mean: " + str(mean))
    print("std dev: " + str(std))
    print("min: " + str(min))
    print("max: " + str(max))

    # evaluate adv samples
    _, test_acc = train_model.evaluate(adv_4_bits, test_labels, verbose=0)
    print("Test accuracy of QNN on adversarial samples: " + str(test_acc))

# train 8 bits
train_graph = tf.Graph()
train_sess = tf.Session(graph=train_graph)

keras.backend.set_session(train_sess)
with train_graph.as_default():
    train_model = build_keras_model()

    # quantization aware training
    #tf.contrib.quantize.create_training_graph(input_graph=train_graph)
    tf.contrib.quantize.experimental_create_training_graph(input_graph=train_graph, weight_bits=8, activation_bits=8)

    train_sess.run(tf.global_variables_initializer())

    train_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    train_model.fit(train_images, train_labels, epochs=2)

    tf.contrib.quantize.experimental_create_eval_graph(input_graph=train_graph, weight_bits=8, activation_bits=8)

    _, test_acc = train_model.evaluate(test_images, test_labels, verbose=0)
    print("Test accuracy of QNN: " + str(test_acc))

    # perform attack on the QNN
    print("Generating adversarial samples for QNN..")
    wrap = KerasModelWrapper(train_model)
    iterative_fgsm = ProjectedGradientDescent(wrap, train_sess)
    adv_8_bits = iterative_fgsm.generate_np(test_images, **FGSM_PARAMS)
    print("Finished generating adversarial samples")

    # quantify perturbation
    mean, std, min, max = get_stats(np.array([np.linalg.norm(x - y) for x, y in zip(test_images, adv_8_bits)]))
    print("Information about L2 distances between adversarial and original samples:")
    print("mean: " + str(mean))
    print("std dev: " + str(std))
    print("min: " + str(min))
    print("max: " + str(max))

    # evaluate adv samples
    _, test_acc = train_model.evaluate(adv_4_bits, test_labels, verbose=0)
    print("Test accuracy of QNN on adversarial samples (4 bits): " + str(test_acc))

    # evaluate adv samples
    _, test_acc = train_model.evaluate(adv_8_bits, test_labels, verbose=0)
    print("Test accuracy of QNN on adversarial samples (8 bits): " + str(test_acc))

# train 16 bits
train_graph = tf.Graph()
train_sess = tf.Session(graph=train_graph)

keras.backend.set_session(train_sess)
with train_graph.as_default():
    train_model = build_keras_model()

    # quantization aware training
    #tf.contrib.quantize.create_training_graph(input_graph=train_graph)
    tf.contrib.quantize.experimental_create_training_graph(input_graph=train_graph, weight_bits=16, activation_bits=16)

    train_sess.run(tf.global_variables_initializer())

    train_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    train_model.fit(train_images, train_labels, epochs=2)

    tf.contrib.quantize.experimental_create_eval_graph(input_graph=train_graph, weight_bits=16, activation_bits=16)

    _, test_acc = train_model.evaluate(test_images, test_labels, verbose=0)
    print("Test accuracy of QNN: " + str(test_acc))

    # perform attack on the QNN
    print("Generating adversarial samples for QNN..")
    wrap = KerasModelWrapper(train_model)
    iterative_fgsm = ProjectedGradientDescent(wrap, train_sess)
    adv_16_bits = iterative_fgsm.generate_np(test_images, **FGSM_PARAMS)
    print("Finished generating adversarial samples")

    # quantify perturbation
    mean, std, min, max = get_stats(np.array([np.linalg.norm(x - y) for x, y in zip(test_images, adv_8_bits)]))
    print("Information about L2 distances between adversarial and original samples:")
    print("mean: " + str(mean))
    print("std dev: " + str(std))
    print("min: " + str(min))
    print("max: " + str(max))

    # evaluate adv samples
    _, test_acc = train_model.evaluate(adv_4_bits, test_labels, verbose=0)
    print("Test accuracy of QNN on adversarial samples (4 bits): " + str(test_acc))

    # evaluate adv samples
    _, test_acc = train_model.evaluate(adv_8_bits, test_labels, verbose=0)
    print("Test accuracy of QNN on adversarial samples (8 bits): " + str(test_acc))

    # evaluate adv samples
    _, test_acc = train_model.evaluate(adv_16_bits, test_labels, verbose=0)
    print("Test accuracy of QNN on adversarial samples (16 bits): " + str(test_acc))
