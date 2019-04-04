from __future__ import print_function
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from NeuralNetwork import *
from DataCollection import *
from upperbound import upperbound

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

network_type = 'full-qnn'
finetune = True


# GPU SETTINGS#
def CudaMemorySettings():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))


def CpuMemorySettings():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    session_conf = tf.ConfigProto(
		device_count={'GPU': 0},
        allow_soft_placement=True,
        log_device_placement=False)
    set_session(tf.Session(config=session_conf))


dataSetName = "mnist"
bound = "ub"
gameType = "cooperative"
image_index = 0
distanceMeasure = "L2"
distance = 10
eta = (distanceMeasure, distance)
tau = 1
wbits = 2
abits = 2
seed = 10

# CudaMemorySettings()
CpuMemorySettings()

nameFile = "seed_" + str(seed) + "_" + str(dataSetName) + "_" + str(image_index) + "_Wbits" + str(
    wbits) + "Abits" + str(abits) + ".txt"

print("name file: " + nameFile)

# calling algorithms
dc = DataCollection("%s_%s_%s_%s_%s_%s_%s" % (dataSetName, bound, tau, gameType, image_index, eta[0], eta[1]))
dc.initialiseIndex(image_index)

print("ok")
(elapsedTime, newConfident, percent, l2dist, l1dist, l0dist, maxFeatures) = (
    upperbound(dataSetName, bound, tau, gameType, image_index, eta, wbits, abits, nameFile, seed))

dc.addRunningTime(elapsedTime)
dc.addConfidence(newConfident)
dc.addManipulationPercentage(percent)
dc.addl2Distance(l2dist)
dc.addl1Distance(l1dist)
dc.addl0Distance(l0dist)
dc.addMaxFeatures(maxFeatures)

dc.provideDetails()
dc.summarise()
dc.close()

K.clear_session()
