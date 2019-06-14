'''
This experiment uses quantization-aware training https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/quantize
provided by the tensorflow lite framework.

The idea is to achieve the same results using the tensorflow lite framework as with the framework used in the rest of this project.
If this turns out to be true, then the current quantization framework is sound. Otherwise, it is important to understand why the results are not the same.

Code base is built on top of https://colab.research.google.com/gist/ohtaman/c1cf119c463fd94b0da50feea320ba1e/edgetpu-with-keras.ipynb#scrollTo=3aluRnFKRN9y
'''


import tensorflow as tf
from tensorflow import keras
from experiments.utils import get_fashion_mnist

print(tf.__version__)

# get dataset
(train_images, train_labels), (test_images, test_labels) = get_fashion_mnist()


def build_keras_model():
    return keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(units=128, activation=tf.nn.relu),
        keras.layers.Dense(units=10, activation=tf.nn.softmax)
    ])

# train
train_graph = tf.Graph()
train_sess = tf.Session(graph=train_graph)

keras.backend.set_session(train_sess)
with train_graph.as_default():
    train_model = build_keras_model()

    # quantization aware training
    tf.contrib.quantize.create_training_graph(input_graph=train_graph)
    train_sess.run(tf.global_variables_initializer())

    train_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    train_model.fit(train_images, train_labels, epochs=1)

    # save graph and checkpoints
    saver = tf.train.Saver()
    saver.save(train_sess, './checkpoints')

    print('sample result of original model')
    print(train_model.predict(test_images[:1]))
    print('ground truth value: ')
    print(test_labels[:1])

# eval
eval_graph = tf.Graph()
eval_sess = tf.Session(graph=eval_graph)

keras.backend.set_session(eval_sess)

with eval_graph.as_default():
    keras.backend.set_learning_phase(0)
    eval_model = build_keras_model()
    tf.contrib.quantize.create_eval_graph(input_graph=eval_graph)
    eval_graph_def = eval_graph.as_graph_def()
    saver = tf.train.Saver()
    saver.restore(eval_sess, 'checkpoints')

    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        eval_sess,
        eval_graph_def,
        [eval_model.output.op.name]
    )

    with open('frozen_model.pb', 'wb') as f:
        f.write(frozen_graph_def.SerializeToString())