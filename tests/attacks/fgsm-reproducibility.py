'''
GIVEN a NN and fixed hyperparameters for the FGSM attack
WHEN the attack is executed twice against the NN
THEN results should be completely the same
'''

SEED_NUMBER_1 = 1
SEED_NUMBER_2 = 2
SEED_NUMBER_3 = 3
import os
os.environ["PYTHONHASHSEED"] = "0"

import numpy as np
import tensorflow as tf
import random as rn
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper
import matplotlib.pyplot as plt


np.random.seed(SEED_NUMBER_1)
rn.seed(SEED_NUMBER_2)

# Force TensorFlow to use single thread.
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
from keras import backend as K
import keras

tf.set_random_seed(SEED_NUMBER_3)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


# prepare dataset
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0

test_images = test_images / 255.0


# train model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=2)

# prepare a sample
img = test_images[rn.randint(0, 100)]
img = (np.expand_dims(img, 0))

# initialize attack
wrap = KerasModelWrapper(model)
fgsm = FastGradientMethod(wrap, sess)
fgsm_params = {'eps': 0.01,
               'clip_min': 0.,
               'clip_max': 1.,
               }

# generate adv samples
adv_1 = fgsm.generate_np(img, **fgsm_params)
adv_2 = fgsm.generate_np(img, **fgsm_params)

print(model.predict(adv_1))
assert np.array_equal(adv_1, adv_2)

# plot original, adversarial for 1st NN, adversarial for 2nd NN, absolute difference
plt.figure(figsize=(1, 4))
diff = abs(adv_1[0] - adv_2[0])

plt.subplot(1, 4, 1)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow(img[0], cmap='gray')

plt.subplot(1, 4, 2)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow(adv_1[0], cmap='gray')

plt.subplot(1, 4, 3)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow(adv_2[0], cmap='gray')

plt.subplot(1, 4, 4)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow(diff, cmap='gray')

plt.show()