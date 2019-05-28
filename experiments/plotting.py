'''
Used for plotting saliency maps.
LINK: https://github.com/raghakot/keras-vis/blob/master/examples/mnist/attention.ipynb

'''

from __future__ import print_function

import numpy as np
import keras

from keras.datasets import mnist
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D

from layers.quantized_layers import QuantizedDense
from layers.quantized_ops import quantized_relu
from matplotlib import pyplot as plt

from vis.visualization import visualize_saliency
from vis.utils import utils
from keras import activations

batch_size = 128
num_classes = 10
epochs = 1

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential()

# QNN
# model.add(keras.layers.Flatten())
# model.add(QuantizedDense(units=128, nb=4, activation=quantized_relu))
# model.add(QuantizedDense(units=10, nb=4, activation=tf.nn.softmax, name='preds'))

# vanilla NN
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation=tf.nn.relu))
model.add(keras.layers.Dense(10, activation=tf.nn.softmax, name='preds'))

# complex vanilla NN
# model = keras.Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=input_shape))
# model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(keras.layers.Dropout(0.25))
# model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(128, activation='relu'))
# model.add(keras.layers.Dropout(0.5))
# model.add(keras.layers.Dense(num_classes, activation='softmax', name='preds'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

class_idx = 0
indices = np.where(y_test[:, class_idx] == 1.)[0]

# pick some random input from here.
# idx = indices[0]
# Lets sanity check the picked image.
# plt.rcParams['figure.figsize'] = (18, 6)
# plt.imshow(x_test[idx][..., 0])


# Utility to search for layer index by name.
# Alternatively we can specify this as -1 since it corresponds to the last layer.
layer_idx = -1

# Swap softmax with linear
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

fig = plt.figure(figsize=(10, 4))
fig.suptitle('Vanilla neural network', fontsize=20)

for class_idx in np.arange(10):
    indices = np.where(y_test[:, class_idx] == 1.)[0]
    idx = indices[0]

    plt.subplot(10, 4, class_idx * 4 + 1)
    plt.imshow(x_test[idx][..., 0], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    for i, modifier in enumerate([None, 'guided', 'relu']):
        grads = visualize_saliency(model, layer_idx, filter_indices=class_idx,
                                   seed_input=x_test[idx], backprop_modifier=modifier)
        if modifier is None:
            modifier = 'vanilla'
        plt.subplot(10, 4, class_idx * 4 + i + 2)
        if class_idx == 0:
            plt.gca().set_title(modifier)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(grads, cmap='jet')
plt.show()
