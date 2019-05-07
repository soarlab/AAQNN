'''
This experiment is structured as follows:
1. Train QNNs for all quantization levels
2. Load samples that are correctly classified by all the QNNs from step 1 (accuracies are 100% on these samples)
3. Run the CW attack for different Q levels
4. Evaluate the QNNs on new adversarial samples
'''

import tensorflow as tf
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.utils_keras import KerasModelWrapper
from keras import backend as K
from experiments.utils import get_fashion_mnist, filter_correctly_classified_samples, get_QNN, get_vanilla_NN
import matplotlib.pyplot as plt
import numpy as np
from experiments.utils import get_stats

EPOCHS = 10
NUMBER_OF_SAMPLES = 1000

CW_PARAMS = {'clip_min': 0.,
             'clip_max': 1.,
             'binary_search_steps': 9,  # number of times to adjust the constant with binary search
             'max_iterations': 10000,  # number of iterations to perform gradient descent
             'abort_early': True,  # if we stop improving, abort gradient descent early
             'learning_rate': 1e-2,  # larger values converge faster to less accurate results
             'initial_const': 1e-3,  # the initial constant c to pick as a first guess
             }

# initialize keras/tf session
sess = tf.Session(graph=tf.get_default_graph())
K.set_session(sess)

# get dataset
(train_images, train_labels), (test_images, test_labels) = get_fashion_mnist()
#(train_images, train_labels), (test_images, test_labels) = get_mnist()

# load models
model_2bits_1 = get_QNN(2)
model_4bits_1 = get_QNN(4)
model_8bits_1 = get_QNN(8)
model_16bits_1 = get_QNN(16)
model_32bits_1 = get_QNN(32)
model_vanilla_nn_1 = get_vanilla_NN()

model_2bits_2 = get_QNN(2)
model_4bits_2 = get_QNN(4)
model_8bits_2 = get_QNN(8)
model_16bits_2 = get_QNN(16)
model_32bits_2 = get_QNN(32)
model_vanilla_nn_2 = get_vanilla_NN()

# train models
print("Training models...")
model_2bits_1.fit(train_images, train_labels, epochs=EPOCHS, verbose=0)
model_4bits_1.fit(train_images, train_labels, epochs=EPOCHS, verbose=0)
model_8bits_1.fit(train_images, train_labels, epochs=EPOCHS, verbose=0)
model_16bits_1.fit(train_images, train_labels, epochs=EPOCHS, verbose=0)
model_32bits_1.fit(train_images, train_labels, epochs=EPOCHS, verbose=0)
model_vanilla_nn_1.fit(train_images, train_labels, epochs=EPOCHS, verbose=0)

model_2bits_2.fit(train_images, train_labels, epochs=EPOCHS, verbose=0)
model_4bits_2.fit(train_images, train_labels, epochs=EPOCHS, verbose=0)
model_8bits_2.fit(train_images, train_labels, epochs=EPOCHS, verbose=0)
model_16bits_2.fit(train_images, train_labels, epochs=EPOCHS, verbose=0)
model_32bits_2.fit(train_images, train_labels, epochs=EPOCHS, verbose=0)
model_vanilla_nn_2.fit(train_images, train_labels, epochs=EPOCHS, verbose=0)
print("Training finished.")

# evaluate models on the test set
_, test_acc = model_2bits_1.evaluate(test_images, test_labels, verbose=0)
print("Test accuracy of QNN_1 with 2 bits: " + str(test_acc))
_, test_acc = model_2bits_2.evaluate(test_images, test_labels, verbose=0)
print("Test accuracy of QNN_2 with 2 bits: " + str(test_acc))

_, test_acc = model_4bits_1.evaluate(test_images, test_labels, verbose=0)
print("Test accuracy of QNN_1 with 4 bits: " + str(test_acc))
_, test_acc = model_4bits_2.evaluate(test_images, test_labels, verbose=0)
print("Test accuracy of QNN_2 with 4 bits: " + str(test_acc))

_, test_acc = model_8bits_1.evaluate(test_images, test_labels, verbose=0)
print("Test accuracy of QNN_1 with 8 bits: " + str(test_acc))
_, test_acc = model_8bits_2.evaluate(test_images, test_labels, verbose=0)
print("Test accuracy of QNN_2 with 8 bits: " + str(test_acc))

_, test_acc = model_16bits_1.evaluate(test_images, test_labels, verbose=0)
print("Test accuracy of QNN_1 with 16 bits: " + str(test_acc))
_, test_acc = model_16bits_2.evaluate(test_images, test_labels, verbose=0)
print("Test accuracy of QNN_2 with 16 bits: " + str(test_acc))

_, test_acc = model_32bits_1.evaluate(test_images, test_labels, verbose=0)
print("Test accuracy of QNN_1 with 32 bits: " + str(test_acc))
_, test_acc = model_32bits_2.evaluate(test_images, test_labels, verbose=0)
print("Test accuracy of QNN_2 with 32 bits: " + str(test_acc))

_, test_acc = model_vanilla_nn_1.evaluate(test_images, test_labels, verbose=0)
print("Test accuracy of vanilla NN_1 (with 32 bits): " + str(test_acc))
_, test_acc = model_vanilla_nn_2.evaluate(test_images, test_labels, verbose=0)
print("Test accuracy of vanilla NN_2 (with 32 bits): " + str(test_acc))


#filter samples correctly classified by all models
all_models = [model_2bits_1, model_2bits_2,
              model_4bits_1, model_4bits_2,
              model_8bits_1, model_8bits_2,
              model_16bits_1, model_16bits_2,
              model_32bits_1, model_32bits_2,
              model_vanilla_nn_1, model_vanilla_nn_2]

test_images, test_labels = filter_correctly_classified_samples(test_images, test_labels, all_models)
print("From now on using " + str(test_images.shape[0]) + " samples that are correctly classified by all " + str(len(all_models)) + " networks.")
print("All neural networks now have 100% accuracy.")
print()

# perform attack on 2 bits QNN
print("Generating adversarial samples for QNN_1 with 2 bits..")
wrap = KerasModelWrapper(model_2bits_1)
cw = CarliniWagnerL2(wrap, sess)
adv = cw.generate_np(test_images, **CW_PARAMS)
print("Finished generating adversarial samples")

# quantify perturbation
mean, std, min, max = get_stats(np.array([np.linalg.norm(x - y) for x, y in zip(test_images, adv)]))
print("Information about L2 distances between adversarial and original samples:")
print("mean: " + str(mean))
print("std dev: " + str(std))
print("min: " + str(min))
print("max: " + str(max))

# evaluate models on adv samples
print("Evaluating accuracy of all neural networks on adversarial samples crafted for 2 bits QNN_1..")

_, test_acc = model_2bits_1.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_1 with 2 bits: " + str(test_acc))
_, test_acc = model_2bits_2.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_2 with 2 bits: " + str(test_acc))

_, test_acc = model_4bits_1.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_1 with 4 bits: " + str(test_acc))
_, test_acc = model_4bits_2.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_2 with 4 bits: " + str(test_acc))

_, test_acc = model_8bits_1.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_1 with 8 bits: " + str(test_acc))
_, test_acc = model_8bits_2.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_2 with 8 bits: " + str(test_acc))

_, test_acc = model_16bits_1.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_1 with 16 bits: " + str(test_acc))
_, test_acc = model_16bits_2.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_2 with 16 bits: " + str(test_acc))

_, test_acc = model_32bits_1.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_1 with 32 bits: " + str(test_acc))
_, test_acc = model_32bits_2.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_2 with 32 bits: " + str(test_acc))

_, test_acc = model_vanilla_nn_1.evaluate(adv, test_labels, verbose=0)
print("Accuracy of vanilla NN_1 (with 32 bits): " + str(test_acc))
_, test_acc = model_vanilla_nn_2.evaluate(adv, test_labels, verbose=0)
print("Accuracy of vanilla NN_2 (with 32 bits): " + str(test_acc))

print()

# perform attack on 4 bits QNN
print("Generating adversarial samples for QNN_1 with 4 bits..")
wrap = KerasModelWrapper(model_4bits_1)
cw = CarliniWagnerL2(wrap, sess)
adv = cw.generate_np(test_images, **CW_PARAMS)
print("Finished generating adversarial samples")

# quantify perturbation
mean, std, min, max = get_stats(np.array([np.linalg.norm(x - y) for x, y in zip(test_images, adv)]))
print("Information about L2 distances between adversarial and original samples:")
print("mean: " + str(mean))
print("std dev: " + str(std))
print("min: " + str(min))
print("max: " + str(max))

# evaluate models on adv samples
print("Evaluating accuracy of all neural networks on adversarial samples crafted for 4 bits QNN_1..")

_, test_acc = model_2bits_1.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_1 with 2 bits: " + str(test_acc))
_, test_acc = model_2bits_2.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_2 with 2 bits: " + str(test_acc))

_, test_acc = model_4bits_1.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_1 with 4 bits: " + str(test_acc))
_, test_acc = model_4bits_2.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_2 with 4 bits: " + str(test_acc))

_, test_acc = model_8bits_1.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_1 with 8 bits: " + str(test_acc))
_, test_acc = model_8bits_2.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_2 with 8 bits: " + str(test_acc))

_, test_acc = model_16bits_1.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_1 with 16 bits: " + str(test_acc))
_, test_acc = model_16bits_2.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_2 with 16 bits: " + str(test_acc))

_, test_acc = model_32bits_1.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_1 with 32 bits: " + str(test_acc))
_, test_acc = model_32bits_2.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_2 with 32 bits: " + str(test_acc))

_, test_acc = model_vanilla_nn_1.evaluate(adv, test_labels, verbose=0)
print("Accuracy of vanilla NN_1 (with 32 bits): " + str(test_acc))
_, test_acc = model_vanilla_nn_2.evaluate(adv, test_labels, verbose=0)
print("Accuracy of vanilla NN_2 (with 32 bits): " + str(test_acc))

print()

# perform attack on 8 bits QNN
print("Generating adversarial samples for QNN_1 with 8 bits..")
wrap = KerasModelWrapper(model_8bits_1)
cw = CarliniWagnerL2(wrap, sess)
adv = cw.generate_np(test_images, **CW_PARAMS)
print("Finished generating adversarial samples")

# quantify perturbation
mean, std, min, max = get_stats(np.array([np.linalg.norm(x - y) for x, y in zip(test_images, adv)]))
print("Information about L2 distances between adversarial and original samples:")
print("mean: " + str(mean))
print("std dev: " + str(std))
print("min: " + str(min))
print("max: " + str(max))

# evaluate models on adv samples
print("Evaluating accuracy of all neural networks on adversarial samples crafted for 8 bits QNN_1..")

_, test_acc = model_2bits_1.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_1 with 2 bits: " + str(test_acc))
_, test_acc = model_2bits_2.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_2 with 2 bits: " + str(test_acc))

_, test_acc = model_4bits_1.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_1 with 4 bits: " + str(test_acc))
_, test_acc = model_4bits_2.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_2 with 4 bits: " + str(test_acc))

_, test_acc = model_8bits_1.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_1 with 8 bits: " + str(test_acc))
_, test_acc = model_8bits_2.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_2 with 8 bits: " + str(test_acc))

_, test_acc = model_16bits_1.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_1 with 16 bits: " + str(test_acc))
_, test_acc = model_16bits_2.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_2 with 16 bits: " + str(test_acc))

_, test_acc = model_32bits_1.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_1 with 32 bits: " + str(test_acc))
_, test_acc = model_32bits_2.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_2 with 32 bits: " + str(test_acc))

_, test_acc = model_vanilla_nn_1.evaluate(adv, test_labels, verbose=0)
print("Accuracy of vanilla NN_1 (with 32 bits): " + str(test_acc))
_, test_acc = model_vanilla_nn_2.evaluate(adv, test_labels, verbose=0)
print("Accuracy of vanilla NN_2 (with 32 bits): " + str(test_acc))

print()

# perform attack on 16 bits QNN
print("Generating adversarial samples for QNN_1 with 16 bits..")
wrap = KerasModelWrapper(model_16bits_1)
cw = CarliniWagnerL2(wrap, sess)
adv = cw.generate_np(test_images, **CW_PARAMS)
print("Finished generating adversarial samples")

# quantify perturbation
mean, std, min, max = get_stats(np.array([np.linalg.norm(x - y) for x, y in zip(test_images, adv)]))
print("Information about L2 distances between adversarial and original samples:")
print("mean: " + str(mean))
print("std dev: " + str(std))
print("min: " + str(min))
print("max: " + str(max))

# evaluate models on adv samples
print("Evaluating accuracy of all neural networks on adversarial samples crafted for 16 bits QNN_1..")

_, test_acc = model_2bits_1.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_1 with 2 bits: " + str(test_acc))
_, test_acc = model_2bits_2.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_2 with 2 bits: " + str(test_acc))

_, test_acc = model_4bits_1.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_1 with 4 bits: " + str(test_acc))
_, test_acc = model_4bits_2.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_2 with 4 bits: " + str(test_acc))

_, test_acc = model_8bits_1.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_1 with 8 bits: " + str(test_acc))
_, test_acc = model_8bits_2.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_2 with 8 bits: " + str(test_acc))

_, test_acc = model_16bits_1.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_1 with 16 bits: " + str(test_acc))
_, test_acc = model_16bits_2.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_2 with 16 bits: " + str(test_acc))

_, test_acc = model_32bits_1.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_1 with 32 bits: " + str(test_acc))
_, test_acc = model_32bits_2.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_2 with 32 bits: " + str(test_acc))

_, test_acc = model_vanilla_nn_1.evaluate(adv, test_labels, verbose=0)
print("Accuracy of vanilla NN_1 (with 32 bits): " + str(test_acc))
_, test_acc = model_vanilla_nn_2.evaluate(adv, test_labels, verbose=0)
print("Accuracy of vanilla NN_2 (with 32 bits): " + str(test_acc))

print()


# perform attack on 32 bits QNN
print("Generating adversarial samples for QNN with 32 bits..")
wrap = KerasModelWrapper(model_32bits_1)
cw = CarliniWagnerL2(wrap, sess)
adv = cw.generate_np(test_images, **CW_PARAMS)
print("Finished generating adversarial samples")

# quantify perturbation
mean, std, min, max = get_stats(np.array([np.linalg.norm(x - y) for x, y in zip(test_images, adv)]))
print("Information about L2 distances between adversarial and original samples:")
print("mean: " + str(mean))
print("std dev: " + str(std))
print("min: " + str(min))
print("max: " + str(max))

# evaluate models on adv samples
print("Evaluating accuracy of all neural networks on adversarial samples crafted for 32 bits QNN_1..")

_, test_acc = model_2bits_1.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_1 with 2 bits: " + str(test_acc))
_, test_acc = model_2bits_2.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_2 with 2 bits: " + str(test_acc))

_, test_acc = model_4bits_1.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_1 with 4 bits: " + str(test_acc))
_, test_acc = model_4bits_2.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_2 with 4 bits: " + str(test_acc))

_, test_acc = model_8bits_1.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_1 with 8 bits: " + str(test_acc))
_, test_acc = model_8bits_2.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_2 with 8 bits: " + str(test_acc))

_, test_acc = model_16bits_1.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_1 with 16 bits: " + str(test_acc))
_, test_acc = model_16bits_2.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_2 with 16 bits: " + str(test_acc))

_, test_acc = model_32bits_1.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_1 with 32 bits: " + str(test_acc))
_, test_acc = model_32bits_2.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_2 with 32 bits: " + str(test_acc))

_, test_acc = model_vanilla_nn_1.evaluate(adv, test_labels, verbose=0)
print("Accuracy of vanilla NN_1 (with 32 bits): " + str(test_acc))
_, test_acc = model_vanilla_nn_2.evaluate(adv, test_labels, verbose=0)
print("Accuracy of vanilla NN_2 (with 32 bits): " + str(test_acc))

print()

# perform attack on (32 bits) vanilla NN
print("Generating adversarial samples for vanilla NN_1 (with 32 bits)..")
wrap = KerasModelWrapper(model_vanilla_nn_1)
cw = CarliniWagnerL2(wrap, sess)
adv = cw.generate_np(test_images, **CW_PARAMS)
print("Finished generating adversarial samples")

# quantify perturbation
mean, std, min, max = get_stats(np.array([np.linalg.norm(x - y) for x, y in zip(test_images, adv)]))
print("Information about L2 distances between adversarial and original samples:")
print("mean: " + str(mean))
print("std dev: " + str(std))
print("min: " + str(min))
print("max: " + str(max))

# evaluate models on adv samples
print("Evaluating accuracy of all neural networks on adversarial samples crafted for vanilla NN_1 (32 bits)..")

_, test_acc = model_2bits_1.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_1 with 2 bits: " + str(test_acc))
_, test_acc = model_2bits_2.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_2 with 2 bits: " + str(test_acc))

_, test_acc = model_4bits_1.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_1 with 4 bits: " + str(test_acc))
_, test_acc = model_4bits_2.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_2 with 4 bits: " + str(test_acc))

_, test_acc = model_8bits_1.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_1 with 8 bits: " + str(test_acc))
_, test_acc = model_8bits_2.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_2 with 8 bits: " + str(test_acc))

_, test_acc = model_16bits_1.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_1 with 16 bits: " + str(test_acc))
_, test_acc = model_16bits_2.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_2 with 16 bits: " + str(test_acc))

_, test_acc = model_32bits_1.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_1 with 32 bits: " + str(test_acc))
_, test_acc = model_32bits_2.evaluate(adv, test_labels, verbose=0)
print("Accuracy of QNN_2 with 32 bits: " + str(test_acc))

_, test_acc = model_vanilla_nn_1.evaluate(adv, test_labels, verbose=0)
print("Accuracy of vanilla NN_1 (with 32 bits): " + str(test_acc))
_, test_acc = model_vanilla_nn_2.evaluate(adv, test_labels, verbose=0)
print("Accuracy of vanilla NN_2 (with 32 bits): " + str(test_acc))

print()

plt.figure(figsize=(5, 5))

for i in range(1, 26):
    plt.subplot(5, 5, i)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(adv[i], cmap='gray')
plt.savefig("CW-vanilla-NN-adv.png")