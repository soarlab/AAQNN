'''
This is expansion of 'FullNNto32bitQNN.py' experiment.

This expeirment is structured as follows:
1. Train 6 NNs with same architectures (32bits CNN, 2,4,8,16,32 bit QNN)
2. Load samples that are correctly classified by all NNs (accuracies are 100% on these samples)
3. Craft adversarial samples for 32 bits CNN out of samples from step 2.
4. Evaluate all networks on samples from step 3.

Except for steps above, see how does the same sample behaves across different quantization levels.
Want to see if there is number of bits after which a sample is not adversarial anymore.

First step is to have a map which saves id of a sample and list of successful attacks per quantization levels.
Second step would be to visualize this. TODO: How to visualize?
'''

'''
This expeirment is structured as follows:
1. Train 2 NNs with same architecture (32bits CNN and 32 bit QNN)
2. Load samples that are correctly classified by both NNs (accuracies are 100% on these samples)
3. Craft adversarial samples for first NN out of samples from step 2.
4. Evaluate first and second NN on the samples from step 3.
'''


import tensorflow as tf
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper
from keras import backend as K
from experiments.utils import get_vanilla_NN, get_scaled_fashion_mnist, filter_correctly_classified_samples, filter_not_correctly_classifed_samples, get_QNN

EPOCHS = 1
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
model_not_quantized = get_vanilla_NN()
model_2bits = get_QNN(2)
model_4bits = get_QNN(4)
model_8bits = get_QNN(8)
model_16bits = get_QNN(16)
model_32bits = get_QNN(32)

# train models
model_not_quantized.fit(train_images, train_labels, epochs=EPOCHS)
model_2bits.fit(train_images, train_labels, epochs=EPOCHS)
model_4bits.fit(train_images, train_labels, epochs=EPOCHS)
model_8bits.fit(train_images, train_labels, epochs=EPOCHS)
model_16bits.fit(train_images, train_labels, epochs=EPOCHS)
model_32bits.fit(train_images, train_labels, epochs=EPOCHS)

# evaluate models on the test set
_, test_acc = model_not_quantized.evaluate(test_images, test_labels)
print("Test accuracy not quantized NN: " + str(test_acc))

_, test_acc = model_2bits.evaluate(test_images, test_labels)
print("Test accuracy of NN with 2 bits: " + str(test_acc))

_, test_acc = model_4bits.evaluate(test_images, test_labels)
print("Test accuracy of NN with 4 bits: " + str(test_acc))

_, test_acc = model_8bits.evaluate(test_images, test_labels)
print("Test accuracy of NN with 8 bits: " + str(test_acc))

_, test_acc = model_16bits.evaluate(test_images, test_labels)
print("Test accuracy of NN with 16 bits: " + str(test_acc))

_, test_acc = model_32bits.evaluate(test_images, test_labels)
print("Test accuracy of NN with 32 bits: " + str(test_acc))


# filter samples correctly classified by both models
all_models = [model_not_quantized, model_2bits, model_4bits, model_8bits, model_16bits, model_32bits]
test_images, test_labels = filter_correctly_classified_samples(test_images, test_labels, all_models)
print("From now on using " + str(test_images.shape[0]) + " samples that are correctly classified by all networks.")
print("All neural networks now have 100% accuracy.")

# perform attack on not quantized
print("Generating adversarial samples for not quantized neural network...")
wrap_1 = KerasModelWrapper(model_not_quantized)
fgsm_1 = FastGradientMethod(wrap_1, sess)
adv = fgsm_1.generate_np(test_images, **FGSM_PARAMS)
print("Finished generating adversarial samples")

# evaluate models on adv samples
print("Evaluating accuracy of all neural networks on adversarial samples crafted for not quantized neural network..")
_, test_acc = model_not_quantized.evaluate(adv, test_labels)
print("Accuracy of not quantized NN: " + str(test_acc))

_, test_acc = model_2bits.evaluate(adv, test_labels)
print("Accuracy of NN with 2 bits: " + str(test_acc))

_, test_acc = model_4bits.evaluate(adv, test_labels)
print("Accuracy of NN with 4 bits: " + str(test_acc))

_, test_acc = model_8bits.evaluate(adv, test_labels)
print("Accuracy of NN with 8 bits: " + str(test_acc))

_, test_acc = model_16bits.evaluate(adv, test_labels)
print("Accuracy of NN with 16 bits: " + str(test_acc))

_, test_acc = model_32bits.evaluate(adv, test_labels)
print("Accuracy of NN with 32 bits: " + str(test_acc))