'''
GIVEN random seed and hyperparameters
WHEN two neural networks are trained
THEN they should be completely the same


In general, following bullets need to be solved:

- Randomness in Initialization, such as weights. (covered here)
- Randomness in Regularization, such as dropout. (covered here)
- Randomness in Optimization, such as stochastic optimization. (covered here)
- Randomness in Training due to parallelization. (solved by using only 1 thread)
- Randomness in Training on GPU due to GPU libraries. (not covered here)
- Randomness in an another library which is using another random generator (not used here)

For more information: https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
...
'''

import os
import numpy as np
import tensorflow as tf
import random as rn
import keras
from keras import backend as K

SEED_NUMBER_1 = 1
SEED_NUMBER_2 = 2
SEED_NUMBER_3 = 3

os.environ["PYTHONHASHSEED"] = "0"


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

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.set_random_seed(SEED_NUMBER_3)

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

model_1.fit(train_images, train_labels, epochs=2)


# reinitialize seeds to same values and train second model
np.random.seed(SEED_NUMBER_1)
rn.seed(SEED_NUMBER_2)
tf.set_random_seed(SEED_NUMBER_3)

model_2 = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model_2.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model_2.fit(train_images, train_labels, epochs=2)


# compare softmax predictions of two models
predictions_1 = model_1.predict(test_images)
predictions_2 = model_2.predict(test_images)
np.testing.assert_array_equal(predictions_1, predictions_2)

# this one is for a user to make user believe that softmax values are same
sample_id = rn.randint(0, 100)
print("predictions 1: ")
print(predictions_1[sample_id])

print("predictions 2: ")
print(predictions_2[sample_id])