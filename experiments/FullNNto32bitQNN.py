'''
This expeirment is structured as follows:
1. Train 2 NNs with same architecture (32bits NN and 32 bit QNN)
2. Load samples that are correctly classified by both NNs (accuracies are 100% on these samples)
3. Craft adversarial samples for first NN out of samples from step 2.
4. Evaluate first and second NN on the samples from step 3.
'''


import tensorflow as tf
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper
from keras import backend as K
from experiments.utils import get_vanilla_NN, get_scaled_fashion_mnist, filter_correctly_classified_samples, filter_not_correctly_classifed_samples, get_QNN

EPOCHS = 10
FGSM_PARAMS = {'eps': 0.05,
               'clip_min': 0.,
               'clip_max': 1.,
               }

# initialize keras/tf session
sess = tf.Session(graph=tf.get_default_graph())
K.set_session(sess)

# get dataset
(train_images, train_labels), (test_images, test_labels) = get_scaled_fashion_mnist()

# load models
model_1 = get_vanilla_NN()
model_2 = get_QNN(32)

# train models
model_1.fit(train_images, train_labels, epochs=EPOCHS, verbose=0)
model_2.fit(train_images, train_labels, epochs=EPOCHS, verbose=0)

# evaluate models on the test set
_, test_acc_1 = model_1.evaluate(test_images, test_labels, verbose=0)
_, test_acc_2 = model_2.evaluate(test_images, test_labels, verbose=0)
print("Test accuracy NN_1: " + str(test_acc_1))
print("Test accuracy NN_2: " + str(test_acc_2))

# filter samples correctly classified by both models
test_images, test_labels = filter_correctly_classified_samples(test_images, test_labels, [model_1, model_2])
print("From now on using " + str(test_images.shape[0]) + " samples that are correctly classified by both networks.")

# assert samples are 100% correctly predicted
_, test_acc_1 = model_1.evaluate(test_images, test_labels, verbose=0)
_, test_acc_2 = model_2.evaluate(test_images, test_labels, verbose=0)

assert test_acc_1 == test_acc_2 == 1.0
print("Both networks have accuracy now 1.0")

# perform attack on NN_1
wrap_1 =  KerasModelWrapper(model_1)
fgsm_1 = FastGradientMethod(wrap_1, sess)
adv = fgsm_1.generate_np(test_images, **FGSM_PARAMS)

# evaluate same adversarial samples on both neural networks
_, adv_test_acc_1 = model_1.evaluate(adv, test_labels, verbose=0)
_, adv_test_acc_2 = model_2.evaluate(adv, test_labels, verbose=0)

# print results
print("Accuracy of NN_1 on adversarial samples crafted for NN_1: " + str(adv_test_acc_1))
print("Accuracy of NN_2 on adversarial samples crafted for NN_1: " + str(adv_test_acc_2))

# filter successful misclassifications on first model
successful_adv_1, correct_labels = filter_not_correctly_classifed_samples(adv, test_labels, [model_1])

# assert correctness of filter
_, zero_acc = model_1.evaluate(successful_adv_1, correct_labels, verbose=0)
assert zero_acc == 0.0

print("Number of samples misclassified by NN_1: ", successful_adv_1.shape[0])

# check how many successful misclassifications from NN 1 transfers to NN 2
successful_adv_2, _ = filter_not_correctly_classifed_samples(successful_adv_1, correct_labels, [model_2])

print("Out of those " + str(successful_adv_1.shape[0]) + " samples, " + str(successful_adv_2.shape[0]) + " are misclassified by NN_2 as well.")