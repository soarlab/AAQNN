'''
Load tf lite (.tflite) model and evaluate it.
'''

import tensorflow as tf
import numpy as np
from experiments.utils import get_fashion_mnist

#Load TFLite model and allocate tensors.
interpreter = tf.contrib.lite.Interpreter(model_path="./model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)

def quantize(detail, data):
    shape = detail['shape']
    dtype = detail['dtype']
    return data.astype(dtype).reshape(shape)


# Test model on random input data.
(train_images, train_labels), (test_images, test_labels) = get_fashion_mnist(scaled=False)

quantized_input = quantize(input_details[0], test_images[4])
interpreter.set_tensor(input_details[0]['index'], quantized_input)

interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print(quantized_input)
print(output_data)
print(test_labels[4])