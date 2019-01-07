# Copyright 2017 Bert Moons

# This file is part of QNN.

# QNN is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# QNN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# The code for QNN is based on BinaryNet: https://github.com/MatthieuCourbariaux/BinaryNet

# You should have received a copy of the GNU General Public License
# along with QNN.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from pylearn2.datasets.cifar10 import CIFAR10
from pylearn2.datasets.mnist import MNIST
import matplotlib.pyplot as plt



def load_dataset(dataset,train_set_size,val_stop):
    if (dataset == "CIFAR-10"):
        '''
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        
        input_shape = (img_rows, img_cols, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        
        y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)
        '''
        print('Loading CIFAR-10 dataset...')

        #train_set.X = np.transpose(np.reshape(np.multiply(255., train_set.X).astype('uint8'), (-1, 1,  28, 28)), (0,2,3,1))

        #train_set_size = 45000
        train_set = CIFAR10(which_set="train", start=0, stop=train_set_size)
        valid_set = CIFAR10(which_set="train", start=train_set_size, stop=val_stop)#50000)
        test_set = CIFAR10(which_set="test")

        train_set.X = np.transpose(np.reshape(train_set.X, (-1, 3, 32, 32)),(0,2,3,1))
        valid_set.X = np.transpose(np.reshape(valid_set.X, (-1, 3, 32, 32)),(0,2,3,1))
        test_set.X = np.transpose(np.reshape(test_set.X, (-1, 3, 32, 32)),(0,2,3,1))
        
        # flatten targets
        train_set.y = np.hstack(train_set.y)
        valid_set.y = np.hstack(valid_set.y)
        test_set.y = np.hstack(test_set.y)

        # Onehot the targets
        train_set.y = np.float32(np.eye(10)[train_set.y])
        valid_set.y = np.float32(np.eye(10)[valid_set.y])
        test_set.y = np.float32(np.eye(10)[test_set.y])

        # for hinge loss
        train_set.y = 2 * train_set.y - 1.
        valid_set.y = 2 * valid_set.y - 1.
        test_set.y = 2 * test_set.y - 1.
        
        # enlarge train data set by mirrroring
        x_train_flip = train_set.X[:, :, ::-1, :]
        y_train_flip = train_set.y
        train_set.X = np.concatenate((train_set.X, x_train_flip), axis=0)
        train_set.y = np.concatenate((train_set.y, y_train_flip), axis=0)

    elif (dataset == "MNIST"):
        print('Loading MNIST dataset...')
        
        #train_set_size = 50000
        #This images are between 0 and 1, 1D array
        train_set = MNIST(which_set="train", start=0, stop=train_set_size, shuffle=False)
        valid_set = MNIST(which_set="train", start=train_set_size, stop=val_stop, shuffle=False)#60000)
        test_set = MNIST(which_set="test", shuffle=False)
        
        #train_set.X = np.multiply(2. / 255., train_set.X)
        #print train_set.X[0]
        #raw_input()
        #train_set.X = np.subtract(train_set.X, 1.)
        #print train_set.X[0]
        #raw_input()
        #train_set.X = np.reshape(train_set.X, (-1, 1, 28, 28))
        #print train_set.X[0]
        #raw_input()
        #train_set.X = np.transpose(train_set.X,(0,2,3,1))
        
        #remove  / 255.  / 255.  / 255.
        #the -1 means check how many elements exists ( I have no time to count them:-))
        # num images x 1 x28x28
        #1 dimension (grey color)
        #28x28x1
        #the transpose is used to obtain num images x 28 x 28 x 1
        
        train_set.X = np.transpose(np.reshape(train_set.X, (-1, 1,  28, 28)), (0,2,3,1))
        valid_set.X = np.transpose(np.reshape(valid_set.X, (-1, 1,  28, 28)), (0,2,3,1))
        test_set.X = np.transpose(np.reshape(test_set.X, (-1, 1,  28, 28)), (0,2,3,1))
        
        # flatten targets
        #From array of one elements to one array for all picture: [1,2,2,8,8,3,...
        
        train_set.y = np.hstack(train_set.y)
        valid_set.y = np.hstack(valid_set.y)
        test_set.y = np.hstack(test_set.y)
        
        # Onehot the targets
        # Each element becomes an array with only one 1 and the rest 0
        train_set.y = np.float32(np.eye(10)[train_set.y])
        valid_set.y = np.float32(np.eye(10)[valid_set.y])
        test_set.y = np.float32(np.eye(10)[test_set.y])
        
        # for hinge loss
        train_set.y = 2 * train_set.y - 1.
        valid_set.y = 2 * valid_set.y - 1.
        test_set.y = 2 * test_set.y - 1.
        
        # enlarge train data set by mirrroring
        x_train_flip = train_set.X[:, :, ::-1, :]
        y_train_flip = train_set.y
        train_set.X = np.concatenate((train_set.X, x_train_flip), axis=0)
        train_set.y = np.concatenate((train_set.y, y_train_flip), axis=0)
        
        #image=test_set.X[800]
        #print image.shape
        #image=test_set.X[800].reshape([28, 28])
        #print test_set.X.shape
        #print train_set.X.shape
        #print valid_set.X.shape
        #plt.figure()
        #plt.imshow(image,cmap='gray')
        #plt.show(block=False)
        #plt.pause(0.05)
        #raw_input()
    else:
        print("wrong dataset given")

    return train_set, valid_set, test_set
