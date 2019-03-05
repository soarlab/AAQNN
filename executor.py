from __future__ import print_function
from keras import backend as K
import sys
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from NeuralNetwork import *
import multiprocessing
from DataCollection import *
from upperbound import upperbound
from lowerbound import lowerbound

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#GPU SETTINGS
# def CudaMemorySettings():
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     config.gpu_options.per_process_gpu_memory_fraction = 0.1
#     config.gpu_options.visible_device_list = "0"
#     set_session(tf.Session(config=config))


def CpuMemorySettings():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    session_conf = tf.ConfigProto(device_count={'GPU' : 0}, allow_soft_placement=True, log_device_placement=False)
    set_session(tf.Session(config=session_conf))

'''
For every image, an adversarial sample is crated for every quantization level (2, 4, .. 64).
Creation of an adversarial sample is isolated in a separate process.
It follows that for every image, there will be 6 different processes.
But for creation of an adversarial sample for the next image, it is not necessary to wait for the previous image to finish.
So the creation of adversarial samples for the next image starts in parallel with the creation of adv samples for the previous image and so on.
'''


def process_image(dataset_name, bound, game_type, image_index, distance_measure, distance, tau, wbits, abits, seed):
    # CudaMemorySettings()
    CpuMemorySettings()

    eta = (distance_measure, distance)
    file_name = "seed_" + str(seed) + "_" + str(dataset_name) + "_" + str(image_index) + "_Wbits" + str(wbits) + "Abits" + str(abits) + ".txt"

    # calling algorithms
    dc = DataCollection("%s_%s_%s_%s_%s_%s_%s" % (dataset_name, bound, tau, game_type, image_index, eta[0], eta[1]))
    dc.initialiseIndex(image_index)

    if bound == 'ub':
        (elapsedTime, newConfident, percent, l2dist, l1dist, l0dist, maxFeatures) = \
            (upperbound(dataset_name, bound, tau, game_type, image_index, eta, wbits, abits, file_name, seed))

        dc.addRunningTime(elapsedTime)
        dc.addConfidence(newConfident)
        dc.addManipulationPercentage(percent)
        dc.addl2Distance(l2dist)
        dc.addl1Distance(l1dist)
        dc.addl0Distance(l0dist)
        dc.addMaxFeatures(maxFeatures)
    elif bound == 'lb':
        lowerbound(dataset_name, image_index, game_type, eta, tau, wbits, abits)

    dc.provideDetails()
    dc.summarise()
    dc.close()
    K.clear_session()


def main():
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    print("pool size: ")
    quantization = [2, 4, 8, 16, 32, 64]

    print(sys.argv[1])
    seed = int(sys.argv[1])
    start_point = int(sys.argv[2])
    test_length = int(sys.argv[3])
    dataset_name = str(sys.argv[4])

    for sourceImage in range(start_point, test_length):
        print(sourceImage)
        index0 = sourceImage
        for q in quantization:
            pool.apply_async(process_image, [dataset_name, "ub", "cooperative", index0, "L2", 10, 1, q, q, seed])

    # wait for all jobs to finish
    pool.close()
    pool.join()
    print("all done")


if __name__=="__main__":
    main()
