'''
This expeirment is structured as follows:
1. Train 2 NNs with same architecture (classic 32bits CNNs)
2. Load samples that are correctly classified by both NNs (accuracies are 100% on these samples)
3. Craft adversarial samples for first NN out of samples from step 2.
4. Evaluate first and second NN on the samples from step 3.
'''

import tensorflow as tf
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper
from keras import backend as K
from experiments import utils

EPOCHS = 1
FGSM_PARAMS = {'eps': 0.05,
               'clip_min': 0.,
               'clip_max': 1.,
               }

# initialize keras/tf session
sess = tf.Session(graph=tf.get_default_graph())
K.set_session(sess)

# get dataset
(train_images, train_labels), (test_images, test_labels) = utils.get_scaled_fashion_mnist()

# load models
model_1 = utils.get_vanilla_CNN()
model_2 = utils.get_vanilla_CNN()

# train models
model_1.fit(train_images, train_labels, epochs=EPOCHS)
model_2.fit(train_images, train_labels, epochs=EPOCHS)

# filter samples correctly classified by both models
test_images, test_labels = utils.filter_positive_samples(test_images, test_labels, [model_1])

# assert samples are 100% correctly predicted
_, test_acc_1 = model_1.evaluate(test_images, test_labels)
print("test acc: ", test_acc_1)
assert test_acc_1 == 1.0
_, test_acc_2 = model_2.evaluate(test_images, test_labels)

assert test_acc_1 == test_acc_2 == 1.0

# perform attack on NN_1
wrap_1 =  KerasModelWrapper(model_1)
fgsm_1 = FastGradientMethod(wrap_1, sess)
adv_1 = fgsm_1.generate_np(test_images, **FGSM_PARAMS)

# evaluate same adversarial samples on both neural networks
_, adv_test_acc_1 = model_1.evaluate(adv_1, test_labels)
_, adv_test_acc_2 = model_2.evaluate(adv_1, test_labels)

# print results
print("Accuracy of original NN on adversarial samples: " + str(adv_test_acc_1))
print("Accuracy of a NN with same architecture as original NN on (same as above) adversarial samples: " + str(adv_test_acc_2))

