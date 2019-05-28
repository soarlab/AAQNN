'''
This experiment is structured as follows:
1. Train QNNs for all quantization levels
2. Load samples that are correctly classified by all the QNNs from step 1 (accuracies are 100% on these samples)
3. Run the iterative FGSM attack for different Q levels
4. Evaluate the QNNs on new adversarial samples

Original paper: https://arxiv.org/pdf/1607.02533.pdf
'''

import tensorflow as tf
from cleverhans.attacks import ProjectedGradientDescent
from cleverhans.utils_keras import KerasModelWrapper
from keras import backend as K
from experiments.utils import get_fashion_mnist, filter_correctly_classified_samples, get_QNN, get_vanilla_NN
import matplotlib.pyplot as plt
import numpy as np
from experiments.utils import get_stats

EPOCHS = 2
EPS = 0.06
FGSM_PARAMS = {'clip_min': 0.,
               'clip_max': 1.,
               'eps': EPS,
               # as in the original paper
               'nb_iter': int(min(EPS * 255 + 4, 1.25 * EPS * 255)),
               'rand_init': 0.
               }

# initialize keras/tf session
sess = tf.Session(graph=tf.get_default_graph())
K.set_session(sess)

# get dataset
(train_images, train_labels), (test_images, test_labels) = get_fashion_mnist()

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
model_vanilla_nn_1.fit(train_images, train_labels, epochs=EPOCHS, verbose=0)
model_2bits_1.fit(train_images, train_labels, epochs=EPOCHS, verbose=0)
model_4bits_1.fit(train_images, train_labels, epochs=EPOCHS, verbose=0)
model_8bits_1.fit(train_images, train_labels, epochs=EPOCHS, verbose=0)
model_16bits_1.fit(train_images, train_labels, epochs=EPOCHS, verbose=0)
model_32bits_1.fit(train_images, train_labels, epochs=EPOCHS, verbose=0)

# plot weights distribution
print ("Vanilla weights")
min_value = None
max_value = None
weights_vanilla = []
for layer in model_vanilla_nn_1.get_weights():
    for neuron in layer:
        if isinstance(neuron, np.float32):
            # bias
            weights_vanilla.append(neuron)
            continue
        for weight in neuron:
            weights_vanilla.append(weight)
            if min_value is None or weight < min_value:
                min_value = weight
            if max_value is None or weight > max_value:
                max_value = weight

ids = [x for x in range(0, len(weights_vanilla))]
plt.scatter(ids, weights_vanilla,  marker=',', s=0.52)
axes = plt.gca()
axes.set_ylim([-1.1,1.1])
plt.title("Vanilla NN")
plt.xlabel('Weight "ids"', fontsize=18)
plt.ylabel('Weight value', fontsize=16)
plt.show()

mean, std, min, max = get_stats(np.array(weights_vanilla))
print("mean: " + str(mean))
print("std dev: " + str(std))
print("min: " + str(min))
print("max: " + str(max))


print ("QNN weights")
min_value = None
max_value = None
weights_qnn = []
for layer in model_8bits_1.get_weights():
    for neuron in layer:
        if isinstance(neuron, np.float32):
            # bias
            weights_qnn.append(neuron)
            continue
        for weight in neuron:
            weights_qnn.append(weight)
            if min_value is None or weight < min_value:
                min_value = weight
            if max_value is None or weight > max_value:
                max_value = weight

plt.scatter(ids, weights_qnn, marker=',', s=0.52)
axes = plt.gca()
axes.set_ylim([-1.1,1.1])
plt.title("QNN")
plt.xlabel('Weight "ids"', fontsize=18)
plt.ylabel('Weight value', fontsize=16)
plt.show()

mean, std, min, max = get_stats(np.array(weights_qnn))
print("mean: " + str(mean))
print("std dev: " + str(std))
print("min: " + str(min))
print("max: " + str(max))


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
iterative_fgsm = ProjectedGradientDescent(wrap, sess)
adv = iterative_fgsm.generate_np(test_images, **FGSM_PARAMS)
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
iterative_fgsm = ProjectedGradientDescent(wrap, sess)
adv = iterative_fgsm.generate_np(test_images, **FGSM_PARAMS)
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
iterative_fgsm = ProjectedGradientDescent(wrap, sess)
adv = iterative_fgsm.generate_np(test_images, **FGSM_PARAMS)
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
iterative_fgsm = ProjectedGradientDescent(wrap, sess)
adv = iterative_fgsm.generate_np(test_images, **FGSM_PARAMS)
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
iterative_fgsm = ProjectedGradientDescent(wrap, sess)
adv = iterative_fgsm.generate_np(test_images, **FGSM_PARAMS)
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
iterative_fgsm = ProjectedGradientDescent(wrap, sess)
adv = iterative_fgsm.generate_np(test_images, **FGSM_PARAMS)
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
plt.savefig("i-fgsm-vanilla-NN-adv" + str(EPS) + ".png")
