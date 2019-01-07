from keras.models import Sequential, Model
from keras import regularizers
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda, concatenate, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
import numpy as np

from layers.quantized_layers import QuantizedConv2D,QuantizedDense
from layers.quantized_ops import quantized_relu as quantize_op
from layers.quantized_ops import quantized_tanh as quantize_tanh_op

from layers.binary_layers import BinaryConv2D
from layers.binary_ops import binary_tanh as binary_tanh_op

def build_model_Deep(cf):
    
    def quantized_relu(x):
        return quantize_op(x,nb=cf.abits)
    
    def quantized_tanh(x):
        return quantize_tanh_op(x,nb=cf.abits)
        
    def binary_tanh(x):
        return binary_tanh_op(x)
    
    H = 1.

    if cf.network_type=='full-qnn':
        Conv_ = lambda s, f, i,c: QuantizedConv2D(kernel_size=(s, s), H=1, nb=cf.wbits, filters=f, strides=(1, 1),
                                            padding='valid', activation='linear',
                                            input_shape = (i,i,c))
        Conv = lambda s, f: QuantizedConv2D(kernel_size=(s, s), H=1, nb=cf.wbits, filters=f, strides=(1, 1),
                                            padding='valid', activation='linear')
        Act = lambda: Activation(quantized_relu)
    else:
        print('wrong network type, the supported network types in this repo are float, qnn, full-qnn, bnn and full-bnn')
    
    model = Sequential()
    model.add(Conv_(3, 32, 28, 1))
    model.add(BatchNormalization(momentum=0.1, epsilon=0.0001))
    model.add(Act())
    
    model.add(Conv (3, 32))
    model.add(BatchNormalization(momentum=0.1, epsilon=0.0001))
    model.add(Act())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv (3, 64))
    model.add(BatchNormalization(momentum=0.1, epsilon=0.0001))
    model.add(Act())

    model.add(Conv (3, 64))
    model.add(BatchNormalization(momentum=0.1, epsilon=0.0001))
    model.add(Act())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(QuantizedDense(units=200,nb=cf.wbits,activation='linear'))
    model.add(BatchNormalization(momentum=0.1, epsilon=0.0001))
    model.add(Act())

    model.add(Dropout(0.5))
    
    model.add(QuantizedDense(units=200,nb=cf.wbits,activation='linear'))
    model.add(BatchNormalization(momentum=0.1, epsilon=0.0001))
    model.add(Act())
    
    model.add(QuantizedDense(units=cf.classes,nb=cf.wbits,activation='softmax'))#THIS LAYER WAS NOT QUANTIZED
            
    model.summary()
    
    return model


def build_model_QNN(cf):
    
    def quantized_relu(x):
        return quantize_op(x,nb=cf.abits)
    
    def quantized_tanh(x):
        return quantize_tanh_op(x,nb=cf.abits)
        
    def binary_tanh(x):
        return binary_tanh_op(x)
    
    H = 1.

    if cf.network_type=='full-qnn':
        Conv_ = lambda s, f, i,c: QuantizedConv2D(kernel_size=(s, s), H=1, nb=cf.wbits, filters=f, strides=(1, 1),
                                            padding='same', activation='linear',
                                            kernel_regularizer=l2(cf.kernel_regularizer),
                                            kernel_lr_multiplier=cf.kernel_lr_multiplier,input_shape = (i,i,c))
        Conv = lambda s, f: QuantizedConv2D(kernel_size=(s, s), H=1, nb=cf.wbits, filters=f, strides=(1, 1),
                                            padding='same', activation='linear',
                                            kernel_regularizer=l2(cf.kernel_regularizer),
                                            kernel_lr_multiplier=cf.kernel_lr_multiplier)
        Act = lambda: Activation(quantized_relu)
    else:
        print('wrong network type, the supported network types in this repo are float, qnn, full-qnn, bnn and full-bnn')
    
    #print cf.nla
    #print cf.nfa
    #print cf.nlb
    #print cf.nfb
    #print cf.nlc
    #print cf.nfc
    #print cf.abits
    #print cf.wbits
    #print cf.network_type
    #raw_input()

    model = Sequential()
    model.add(Conv_(3, cf.nfa,cf.dim,cf.channels))
    model.add(BatchNormalization(momentum=0.1,epsilon=0.0001))
    model.add(Act())
    
    # block A
    for i in range(0,cf.nla-1):
        model.add(Conv(3, cf.nfa))
        model.add(BatchNormalization(momentum=0.1, epsilon=0.0001))
        model.add(Act())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # block B
    for i in range(0,cf.nlb):
        model.add(Conv(3, cf.nfb))
        model.add(BatchNormalization(momentum=0.1, epsilon=0.0001))
        model.add(Act())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # block C
    for i in range(0,cf.nlc):
        model.add(Conv(3, cf.nfc))
        model.add(BatchNormalization(momentum=0.1, epsilon=0.0001))
        model.add(Act())
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Dense Layer
    model.add(Flatten())
    model.add(QuantizedDense(units=cf.classes,nb=cf.wbits,use_bias=False))
    model.add(BatchNormalization(momentum=0.1,epsilon=0.0001))

    model.summary()
    return model


def load_weights_QNN(model, weight_reader):
    weight_reader.reset()

    for i in range(len(model.layers)):
        if 'conv' in model.layers[i].name:
            if 'batch' in model.layers[i + 1].name:
                norm_layer = model.layers[i + 1]
                size = np.prod(norm_layer.get_weights()[0].shape)

                beta = weight_reader.read_bytes(size)
                gamma = weight_reader.read_bytes(size)
                mean = weight_reader.read_bytes(size)
                var = weight_reader.read_bytes(size)

                weights = norm_layer.set_weights([gamma, beta, mean, var])

            conv_layer = model.layers[i]
            if len(conv_layer.get_weights()) > 1:
                bias = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
                kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                kernel = kernel.transpose([2, 3, 1, 0])
                conv_layer.set_weights([kernel, bias])
            else:
                kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                kernel = kernel.transpose([2, 3, 1, 0])
                conv_layer.set_weights([kernel])
    return model
