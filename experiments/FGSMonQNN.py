'''
This experiment is structured as follows:
1. Train QNNs for all quantization levels
2. Load samples that are correctly classified by all the QNNs from step 1 (accuracies are 100% on these samples)
3. Run the FGSM attack for different Q levels
4. Evaluate the QNNs on new adversarial samples
'''

import tensorflow as tf
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper
from keras import backend as K
from experiments.utils import get_scaled_fashion_mnist, filter_correctly_classified_samples, get_QNN

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
model_2bits = get_QNN(2)
model_4bits = get_QNN(4)
model_8bits = get_QNN(8)
model_16bits = get_QNN(16)
model_32bits = get_QNN(32)

# train models
print("Training models...")
model_2bits.fit(train_images, train_labels, epochs=EPOCHS, verbose=0)
model_4bits.fit(train_images, train_labels, epochs=EPOCHS, verbose=0)
model_8bits.fit(train_images, train_labels, epochs=EPOCHS, verbose=0)
model_16bits.fit(train_images, train_labels, epochs=EPOCHS, verbose=0)
model_32bits.fit(train_images, train_labels, epochs=EPOCHS, verbose=0)
print("Training finished.")

# evaluate models on the test set
_, test_acc = model_2bits.evaluate(test_images, test_labels, verbose=0)
print("Test accuracy of NN with 2 bits: " + str(test_acc))

_, test_acc = model_4bits.evaluate(test_images, test_labels, verbose=0)
print("Test accuracy of NN with 4 bits: " + str(test_acc))

_, test_acc = model_8bits.evaluate(test_images, test_labels, verbose=0)
print("Test accuracy of NN with 8 bits: " + str(test_acc))

_, test_acc = model_16bits.evaluate(test_images, test_labels, verbose=0)
print("Test accuracy of NN with 16 bits: " + str(test_acc))

_, test_acc = model_32bits.evaluate(test_images, test_labels, verbose=0)
print("Test accuracy of NN with 32 bits: " + str(test_acc))


# filter samples correctly classified by both models
all_models = [model_2bits, model_4bits, model_8bits, model_16bits, model_32bits]
test_images, test_labels = filter_correctly_classified_samples(test_images, test_labels, all_models)
print("From now on using " + str(test_images.shape[0]) + " samples that are correctly classified by all networks.")
print("All neural networks now have 100% accuracy.")

# perform attack on 2 bits QNN
print("Generating adversarial samples for QNN with 2 bits..")
wrap = KerasModelWrapper(model_2bits)
fgsm = FastGradientMethod(wrap, sess)
adv = fgsm.generate_np(test_images, **FGSM_PARAMS)
print("Finished generating adversarial samples")

# evaluate models on adv samples
print("Evaluating accuracy of all neural networks on adversarial samples crafted for 2 bits QNN..")

_, test_acc = model_2bits.evaluate(adv, test_labels, verbose=0)
print("Accuracy of NN with 2 bits: " + str(test_acc))

_, test_acc = model_4bits.evaluate(adv, test_labels, verbose=0)
print("Accuracy of NN with 4 bits: " + str(test_acc))

_, test_acc = model_8bits.evaluate(adv, test_labels, verbose=0)
print("Accuracy of NN with 8 bits: " + str(test_acc))

_, test_acc = model_16bits.evaluate(adv, test_labels, verbose=0)
print("Accuracy of NN with 16 bits: " + str(test_acc))

_, test_acc = model_32bits.evaluate(adv, test_labels, verbose=0)
print("Accuracy of NN with 32 bits: " + str(test_acc))

print()

# perform attack on 4 bits QNN
print("Generating adversarial samples for QNN with 4 bits..")
wrap = KerasModelWrapper(model_4bits)
fgsm = FastGradientMethod(wrap, sess)
adv = fgsm.generate_np(test_images, **FGSM_PARAMS)
print("Finished generating adversarial samples")

# evaluate models on adv samples
print("Evaluating accuracy of all neural networks on adversarial samples crafted for 4 bits QNN..")

_, test_acc = model_2bits.evaluate(adv, test_labels, verbose=0)
print("Accuracy of NN with 2 bits: " + str(test_acc))

_, test_acc = model_4bits.evaluate(adv, test_labels, verbose=0)
print("Accuracy of NN with 4 bits: " + str(test_acc))

_, test_acc = model_8bits.evaluate(adv, test_labels, verbose=0)
print("Accuracy of NN with 8 bits: " + str(test_acc))

_, test_acc = model_16bits.evaluate(adv, test_labels, verbose=0)
print("Accuracy of NN with 16 bits: " + str(test_acc))

_, test_acc = model_32bits.evaluate(adv, test_labels, verbose=0)
print("Accuracy of NN with 32 bits: " + str(test_acc))

print()

# perform attack on 8 bits QNN
print("Generating adversarial samples for QNN with 8 bits..")
wrap = KerasModelWrapper(model_8bits)
fgsm = FastGradientMethod(wrap, sess)
adv = fgsm.generate_np(test_images, **FGSM_PARAMS)
print("Finished generating adversarial samples")

# evaluate models on adv samples
print("Evaluating accuracy of all neural networks on adversarial samples crafted for 8 bits QNN..")

_, test_acc = model_2bits.evaluate(adv, test_labels, verbose=0)
print("Accuracy of NN with 2 bits: " + str(test_acc))

_, test_acc = model_4bits.evaluate(adv, test_labels, verbose=0)
print("Accuracy of NN with 4 bits: " + str(test_acc))

_, test_acc = model_8bits.evaluate(adv, test_labels, verbose=0)
print("Accuracy of NN with 8 bits: " + str(test_acc))

_, test_acc = model_16bits.evaluate(adv, test_labels, verbose=0)
print("Accuracy of NN with 16 bits: " + str(test_acc))

_, test_acc = model_32bits.evaluate(adv, test_labels, verbose=0)
print("Accuracy of NN with 32 bits: " + str(test_acc))

print()

# perform attack on 16 bits QNN
print("Generating adversarial samples for QNN with 16 bits..")
wrap = KerasModelWrapper(model_16bits)
fgsm = FastGradientMethod(wrap, sess)
adv = fgsm.generate_np(test_images, **FGSM_PARAMS)
print("Finished generating adversarial samples")

# evaluate models on adv samples
print("Evaluating accuracy of all neural networks on adversarial samples crafted for 16 bits QNN..")

_, test_acc = model_2bits.evaluate(adv, test_labels, verbose=0)
print("Accuracy of NN with 2 bits: " + str(test_acc))

_, test_acc = model_4bits.evaluate(adv, test_labels, verbose=0)
print("Accuracy of NN with 4 bits: " + str(test_acc))

_, test_acc = model_8bits.evaluate(adv, test_labels, verbose=0)
print("Accuracy of NN with 8 bits: " + str(test_acc))

_, test_acc = model_16bits.evaluate(adv, test_labels, verbose=0)
print("Accuracy of NN with 16 bits: " + str(test_acc))

_, test_acc = model_32bits.evaluate(adv, test_labels, verbose=0)
print("Accuracy of NN with 32 bits: " + str(test_acc))

print()


# perform attack on 32 bits QNN
print("Generating adversarial samples for QNN with 32 bits..")
wrap = KerasModelWrapper(model_32bits)
fgsm = FastGradientMethod(wrap, sess)
adv = fgsm.generate_np(test_images, **FGSM_PARAMS)
print("Finished generating adversarial samples")

# evaluate models on adv samples
print("Evaluating accuracy of all neural networks on adversarial samples crafted for 32 bits QNN..")

_, test_acc = model_2bits.evaluate(adv, test_labels, verbose=0)
print("Accuracy of NN with 2 bits: " + str(test_acc))

_, test_acc = model_4bits.evaluate(adv, test_labels, verbose=0)
print("Accuracy of NN with 4 bits: " + str(test_acc))

_, test_acc = model_8bits.evaluate(adv, test_labels, verbose=0)
print("Accuracy of NN with 8 bits: " + str(test_acc))

_, test_acc = model_16bits.evaluate(adv, test_labels, verbose=0)
print("Accuracy of NN with 16 bits: " + str(test_acc))

_, test_acc = model_32bits.evaluate(adv, test_labels, verbose=0)
print("Accuracy of NN with 32 bits: " + str(test_acc))

print()
