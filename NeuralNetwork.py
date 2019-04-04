"""
Construct a NeuralNetwork class to include operations
related to various datasets and corresponding models.

Author: Min Wu
Email: min.wu@cs.ox.ac.uk
"""

from keras.models import Sequential
from keras.preprocessing import image as Image
from keras import backend as K
from matplotlib import pyplot as plt
from utils.config_utils import Config

from basics import assure_path_exists
from DataSet import *
import func
import numpy


# Define a Neural Network class.
class NeuralNetwork:
    class myCF:
        myDict = {}

        def __init__(self, nn):
            self.myDict['abits'] = nn.abits
            self.myDict['wbits'] = nn.wbits
            self.myDict['network_type'] = nn.network_type
            self.myDict['seed'] = nn.seed

    # Specify which dataset at initialisation.
    def __init__(self, data_set, abits, wbits, network_type, seed):
        self.network_type = network_type
        self.abits = abits
        self.wbits = wbits
        self.data_set = data_set
        self.seed = seed
        self.model = Sequential()
        cfDeep = self.myCF(self)
        if self.data_set == 'mnist':
            cfg = 'config_MNIST'
        if self.data_set == 'fashion':
            cfg = 'config_FASHION'
        if self.data_set == 'cifar10':
            cfg = 'config_CIFAR-10'
        self.cf = Config(cfg, cmd_args=cfDeep.myDict)
        print("Dataset: " + str("%s_pic/" % self.data_set))
        assure_path_exists("%s_pic/" % self.data_set)

    def predict(self, image):
        import numpy as np
        image = np.expand_dims(image, axis=0)
        predict_value = self.model.predict(image)
        new_class = np.argmax(np.ravel(predict_value))
        confident = np.amax(np.ravel(predict_value))
        return new_class, confident

    # To train a neural network.
    def train_network_QNN(self):
        # Train an mnist model.

        if self.data_set == 'mnist':
            datasetTrain = DataSet('mnist', 'training')
            datasetTest = DataSet('mnist', 'test')
        if self.data_set == 'cifar10':
            datasetTrain = DataSet('cifar10', 'training')
            datasetTest = DataSet('cifar10', 'test')
        if self.data_set == 'fashion':
            datasetTrain = DataSet('fashion', 'training')
            datasetTest = DataSet('fashion', 'test')

        train_x, train_y = datasetTrain.get_dataset()
        test_x, test_y = datasetTest.get_dataset()

        needToTrain, myModel = func.getModelFromQNN(self.cf, train_x, train_y, test_x, test_y)

        # myModel=func.getModelFromDeepGame(cf, train_x,train_y,test_x,test_y,epochs,batch_size)

        self.model = myModel

        score = (self.model).evaluate(test_x, test_y, verbose=0)
        print("Precision " + str(self.abits) + " " + str(self.wbits) + " Test loss:", score[0])
        print("Precision " + str(self.abits) + " " + str(self.wbits) + " Test accuracy:", score[1])

    def save_input(self, image, filename):
        image = Image.array_to_img(image.copy())
        plt.imsave(filename, image)
        # causes discrepancy
        # image_cv = copy.deepcopy(image)
        # cv2.imwrite(filename, image_cv * 255.0, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    def get_label(self, index):
        if self.data_set == 'mnist':
            labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        elif self.data_set == 'fashion':
            labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        elif self.data_set == 'cifar10':
            labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        elif self.data_set == 'gtsrb':
            labels = ['speed limit 20 (prohibitory)', 'speed limit 30 (prohibitory)',
                      'speed limit 50 (prohibitory)', 'speed limit 60 (prohibitory)',
                      'speed limit 70 (prohibitory)', 'speed limit 80 (prohibitory)',
                      'restriction ends 80 (other)', 'speed limit 100 (prohibitory)',
                      'speed limit 120 (prohibitory)', 'no overtaking (prohibitory)',
                      'no overtaking (trucks) (prohibitory)', 'priority at next intersection (danger)',
                      'priority road (other)', 'give way (other)', 'stop (other)',
                      'no traffic both ways (prohibitory)', 'no trucks (prohibitory)',
                      'no entry (other)', 'danger (danger)', 'bend left (danger)',
                      'bend right (danger)', 'bend (danger)', 'uneven road (danger)',
                      'slippery road (danger)', 'road narrows (danger)', 'construction (danger)',
                      'traffic signal (danger)', 'pedestrian crossing (danger)', 'school crossing (danger)',
                      'cycles crossing (danger)', 'snow (danger)', 'animals (danger)',
                      'restriction ends (other)', 'go right (mandatory)', 'go left (mandatory)',
                      'go straight (mandatory)', 'go right or straight (mandatory)',
                      'go left or straight (mandatory)', 'keep right (mandatory)',
                      'keep left (mandatory)', 'roundabout (mandatory)',
                      'restriction ends (overtaking) (other)', 'restriction ends (overtaking (trucks)) (other)']
        else:
            print("LABELS: Unsupported dataset.")
        return labels[index]

    def getImageNumber(self, targetLabel):
        if self.data_set == 'mnist':
            datasetTest = DataSet('mnist', 'test')
        if self.data_set == 'cifar10':
            datasetTest = DataSet('cifar10', 'test')
        test_x, test_y = datasetTest.get_dataset()

        myList = []
        for i in range(0, 10000):
            label = numpy.where(test_y[i] > 0)[0][0]
            strLabel = self.get_label(int(label))
            if str(targetLabel) == str(strLabel):
                myList.append(i)
        print(myList)
        raw_input()

    # Get softmax logits, i.e., the inputs to the softmax function of the classification layer,
    # as softmax probabilities may be too close to each other after just one pixel manipulation.
    def softmax_logits(self, manipulated_images, batch_size=512):
        model = self.model

        func = K.function([model.layers[0].input] + [K.learning_phase()],
                          [model.layers[model.layers.__len__() - 1].output.op.inputs[0]])

        batch, remainder = divmod(len(manipulated_images), batch_size)


        if len(manipulated_images) >= batch_size:
            softmax_logits = []
            for b in range(batch):
                logits = func([manipulated_images[b * batch_size:(b + 1) * batch_size], 0])[0]
                softmax_logits.append(logits)
            softmax_logits = np.asarray(softmax_logits)
            softmax_logits = softmax_logits.reshape(batch * batch_size, model.output_shape[1])
            # note that here if logits is empty, it is fine, as it won't be concatenated.
            if batch * batch_size < len(manipulated_images):
                logits = func([manipulated_images[batch * batch_size:len(manipulated_images)], 0])[0]
                softmax_logits = np.concatenate((softmax_logits, logits), axis=0)
        else:
            softmax_logits = func([manipulated_images, 0])[0]

        return softmax_logits
