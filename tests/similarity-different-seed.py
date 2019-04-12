'''
GIVEN enough number of epochs for training
WHEN two neural networks with same architectures are trained (using the different seed)
THEN they should have similar accuracy

For seed manipulation, check "reproducibility-same-seed.py" test
'''

# 1% difference allowed in accuracy
SIMILARITY_VALUE = 0.01

SEED_NUMBER_1 = 1
SEED_NUMBER_2 = 2
SEED_NUMBER_3 = 3
import os
os.environ["PYTHONHASHSEED"] = "0"

import numpy as np
import tensorflow as tf
import random as rn

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(SEED_NUMBER_1)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rn.seed(SEED_NUMBER_2)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
import keras
from keras import backend as K

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.set_random_seed(SEED_NUMBER_3)

EPOCHS_NUMBER = 10

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


# prepare dataset
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0

test_images = test_images / 255.0


# train first model
model_1 = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model_1.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model_1.fit(train_images, train_labels, epochs=EPOCHS_NUMBER)


# reinitialize seeds to different values and train second model
np.random.seed(SEED_NUMBER_1 + 1)
rn.seed(SEED_NUMBER_2 + 1)
tf.set_random_seed(SEED_NUMBER_3 + 1)

model_2 = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model_2.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model_2.fit(train_images, train_labels, epochs=EPOCHS_NUMBER)


# evaluate two models
_, test_acc_1 = model_1.evaluate(test_images, test_labels)
_, test_acc_2 = model_2.evaluate(test_images, test_labels)

assert test_acc_2 * (1 - SIMILARITY_VALUE) <= test_acc_1 <= test_acc_2 * (1 + SIMILARITY_VALUE)