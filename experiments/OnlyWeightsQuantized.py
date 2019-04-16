'''
This expeirment is structured as follows:
1. Train QNN and save weights
2. Load weights of QNN in normal 32bits NN; this NN doesn't know anything about quantization
3. Repeat steps 1 and 2 for all Q levels.
4. Load samples that are correctly classified by all NNs (accuracies are 100% on these samples)
5. Craft adversarial samples for 32 bits NNs out of samples from step 4.
6. Evaluate the NNs on new adversarial samples
'''

import tensorflow as tf
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper
from keras import backend as K
from experiments.utils import get_vanilla_NN, get_scaled_fashion_mnist, filter_correctly_classified_samples, get_QNN


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

# train models
model_2bits.fit(train_images, train_labels, epochs=EPOCHS, verbose=0)

_, test_acc = model_2bits.evaluate(test_images, test_labels, verbose=0)
print("Test accuracy of NN with 2 bits: " + str(test_acc))

# save weights
print("Saving weights")
model_2bits.save_weights("2bits-weights.h5")

# load not quantized model and quantized weights
print("Loading the weights in same architecture NN but no quantization involved")
model = get_vanilla_NN()
model.load_weights("2bits-weights.h5")

_, test_acc = model.evaluate(test_images, test_labels, verbose=0)
print("Test accuracy of NN with weights of 2 bits but no quantization involved: " + str(test_acc))

# load quantized
print("Loading the weights in same architecture NN with 2 bits quantization")
model = get_QNN(2)
model.load_weights("2bits-weights.h5")

_, test_acc = model.evaluate(test_images, test_labels, verbose=0)
print("Test accuracy of NN with weights of 2 bits with quantization involved: " + str(test_acc))

print("Stopping the experiment due to too low accuracy (no point in attacking the network that is not performing well).")
print("Maybe because activation functions (and weights) are constrained to -1 and +1 in QNNs ? ")
