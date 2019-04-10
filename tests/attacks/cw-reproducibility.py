'''
GIVEN a NN and fixed hyperparameters for the CW attack
WHEN the attack is executed twice against the same NN
THEN results should be completely the same
'''
import numpy as np
import tensorflow as tf
import random as rn
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.utils_keras import KerasModelWrapper
import matplotlib.pyplot as plt
from keras import backend as K
import keras


cw_params = {'initial_const': 10}

sess = tf.Session(graph=tf.get_default_graph())
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
cw = CarliniWagnerL2(wrap, sess)

# generate adv samples
adv_1 = cw.generate_np(img, **cw_params)
adv_2 = cw.generate_np(img, **cw_params)

print(model.predict(adv_1))
assert np.array_equal(adv_1, adv_2)

# plot original, first attack, second attack, absolute difference
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