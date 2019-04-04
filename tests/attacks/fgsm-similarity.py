'''
GIVEN fixed hyperparameters for the FGSM attack and enough number of epochs for training
WHEN two neural networks with same architectures are trained (using the different seed)
THEN results of attacks should be completely (or almost) the same
'''
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper

# 3% difference allowed in accuracy (e.g. 83% and 79% accuracies are ok)
ACCURACY_SIMILARITY = 0.03

# 1% in perturbation difference allowed 
PERTURBATION_SIMILARITY = 0.01

# fixed hyperparameters
FGSM_PARAMS = {'eps': 0.1,
               'clip_min': 0.,
               'clip_max': 1.,
               }
EPOCHS_NUMBER = 10


def get_new_instance_of_trained_NN(train_images, train_labels):
    # train first model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=EPOCHS_NUMBER)
    return model


def get_stats(values):
    return np.mean(values), np.std(values), np.min(values), np.max(values)


# initialize keras/tf session
sess = tf.Session(graph=tf.get_default_graph())
K.set_session(sess)

# prepare dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

# prepare models
model_1 = get_new_instance_of_trained_NN(train_images, train_labels)
model_2 = get_new_instance_of_trained_NN(train_images, train_labels)

# assert similar accuracy against legit samples
_, test_acc_1 = model_1.evaluate(test_images, test_labels)
_, test_acc_2 = model_2.evaluate(test_images, test_labels)
assert abs(test_acc_2 - test_acc_1) < ACCURACY_SIMILARITY

# perform attacks
wrap_1 = KerasModelWrapper(model_1)
fgsm_1 = FastGradientMethod(wrap_1, sess)
adv_1 = fgsm_1.generate_np(test_images, **FGSM_PARAMS)

wrap_2 = KerasModelWrapper(model_2)
fgsm_2 = FastGradientMethod(wrap_2, sess)
adv_2 = fgsm_2.generate_np(test_images, **FGSM_PARAMS)

# assert similar accuracy against adv samples
_, adv_test_acc_1 = model_1.evaluate(adv_1, test_labels)
_, adv_test_acc_2 = model_2.evaluate(adv_2, test_labels)

assert abs(adv_test_acc_2 - adv_test_acc_1) < ACCURACY_SIMILARITY

# compute difference between legit and adv samples (L1 norm)
diff_1 = test_images - adv_1
diff_2 = test_images - adv_2

# assert differences in perturbations are similar enough
mean_1, std_1, min_1, max_1 = get_stats(diff_1)
mean_2, std_2, min_2, max_2 = get_stats(diff_2)

assert abs(mean_2 - mean_1) < abs(PERTURBATION_SIMILARITY * mean_1)
assert abs(std_2 - std_1) < abs(PERTURBATION_SIMILARITY * std_1)
assert abs(min_2 - min_1) < abs(PERTURBATION_SIMILARITY * min_1)
assert abs(max_2 - max_1) < abs(PERTURBATION_SIMILARITY * max_1)


# print results
print("legit accuracy 1: " + str(test_acc_1))
print("legit accuracy 2: " + str(test_acc_2))
print()

print("adv accuracy 1: " + str(adv_test_acc_1))
print("adv accuracy 2: " + str(adv_test_acc_2))
print()

print("mean 1: " + str(mean_1))
print("std 1: " + str(std_1))
print("min 1: " + str(min_1))
print("max 1: " + str(max_1))
print()

print("mean 2: " + str(mean_2))
print("std 2: " + str(std_2))
print("min 2: " + str(min_2))
print("max 2: " + str(max_2))
