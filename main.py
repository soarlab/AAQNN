from __future__ import print_function
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from NeuralNetwork import *
from DataCollection import *
from upperbound import upperbound
from lowerbound import lowerbound

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

network_type = 'full-qnn'
finetune = True


# GPU SETTINGS#
def CudaMemorySettings():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))


def CpuMemorySettings():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    session_conf = tf.ConfigProto(
        device_count={'GPU': 0},
        allow_soft_placement=True,
        log_device_placement=False
    )
    set_session(tf.Session(config=session_conf))


def validate_arguments(dataset_name,
                       bound,
                       game_type,
                       image_index,
                       distance_measure,
                       distance,
                       tau,
                       wbits,
                       abits,
                       seed):
    if dataset_name != 'mnist' and dataset_name != 'cifar10' and dataset_name != 'fashion':
        print("please specify the dataset: mnist or cifar10 or fashion")
        exit(1)

    if bound != 'ub' and bound != 'lb':
        print("please specify the bound as : ub or lb")
        exit(1)

    if game_type != 'cooperative' and game_type != 'competitive':
        print("please specify the game type as: cooperative or competitive")
        exit(1)

    if not isinstance(image_index, int):
        print("please specify the index of the image as type [int]")
        exit(1)

    if distance_measure != 'L0' and distance_measure != 'L1' and distance_measure != 'L2':
        print("please specify the distance measure as: L0, L1, or L2")
        exit(1)

    if not (isinstance(distance, float) or isinstance(distance, int)):
        print("please specify the distance as type [int/float]")
        exit(1)

    if not (isinstance(tau, float) or isinstance(tau, int)):
        print("please specify the tau as type [int/float]")
        exit(1)

    if not isinstance(wbits, int):
        print("please specify the wbits as type [int]")
        exit(1)

    if not isinstance(abits, int):
        print("please specify the wbits as type [int]")
        exit(1)

    if not isinstance(seed, int):
        print("please specify the seed as type [int]")
        exit(1)

    print("All arguments validated successfully")


def process_image(dataset_name,
                  bound,
                  game_type,
                  image_index,
                  distance_measure,
                  distance,
                  tau,
                  wbits,
                  abits,
                  seed):
    validate_arguments(dataset_name, bound, game_type, image_index,
                       distance_measure, distance, tau, wbits, abits, seed)
    print("OK")

    eta = (distance_measure, distance)

    print("Seed:" + str(seed))

    # CudaMemorySettings()
    CpuMemorySettings()

    file_name = "seed_" + str(seed) + \
                "_" + str(dataset_name) + \
                "_" + str(image_index) + \
                "_Wbits" + str(wbits) + \
                "Abits" + str(abits) + \
                ".txt"

    # calling algorithms
    dc = DataCollection("%s_%s_%s_%s_%s_%s_%s" % (dataset_name, bound, tau, game_type, image_index, eta[0], eta[1]))
    dc.initialiseIndex(image_index)

    if bound == 'ub':
        (elapsedTime, newConfident, percent, l2dist, l1dist, l0dist, maxFeatures) = (
            upperbound(dataset_name, bound, tau, game_type, image_index, eta, wbits, abits, file_name, seed))

        dc.addRunningTime(elapsedTime)
        dc.addConfidence(newConfident)
        dc.addManipulationPercentage(percent)
        dc.addl2Distance(l2dist)
        dc.addl1Distance(l1dist)
        dc.addl0Distance(l0dist)
        dc.addMaxFeatures(maxFeatures)

    elif bound == 'lb':
        lowerbound(dataset_name, image_index, game_type, eta, tau, wbits, abits)

    else:
        print("lower bound algorithm is developing...")
        exit(1)

    dc.provideDetails()
    dc.summarise()
    dc.close()

    K.clear_session()
