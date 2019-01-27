from __future__ import print_function
from keras import backend as K
import sys
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from NeuralNetwork import *
from DataSet import *
from DataCollection import *
from upperbound import upperbound
from lowerbound import lowerbound

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

network_type='full-qnn'
finetune=True

#GPU SETTINGS#
def CudaMemorySettings():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))

def CpuMemorySettings():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    session_conf = tf.ConfigProto(
    device_count={'GPU' : 0},
    allow_soft_placement=True,
    log_device_placement=False)
    set_session(tf.Session(config=session_conf))

print ("OK")
if len(sys.argv) == 11:
	if sys.argv[1] == 'mnist' or sys.argv[1] == 'cifar10' or sys.argv[1] == 'fashion':
			dataSetName = sys.argv[1]
	else:
			print("please specify as the 1st argument the dataset: mnist or cifar10 or gtsrb")
			exit(1)
	if sys.argv[2] == 'ub' or sys.argv[2] == 'lb':
			bound = sys.argv[2]
	else:
			print("please specify as the 2nd argument the bound: ub or lb")
			exit(1)
	
	if sys.argv[3] == 'cooperative' or sys.argv[3] == 'competitive':
			gameType = sys.argv[3]
	else:
			print("please specify as the 3nd argument the game mode: cooperative or competitive")
			exit(1)
	
	if isinstance(int(sys.argv[4]), int):
			image_index = int(sys.argv[4])
	else:
			print("please specify as the 4th argument the index of the image: [int]")
			exit(1)
	
	if sys.argv[5] == 'L0' or sys.argv[5] == 'L1' or sys.argv[5] == 'L2':
			distanceMeasure = sys.argv[5]
	else:
			print("please specify as the 5th argument the distance measure: L0, L1, or L2")
			exit(1)
	
	if isinstance(float(sys.argv[6]), float):
			distance = float(sys.argv[6])
	else:
			print("please specify as the 6th argument the distance: [int/float]")
			exit(1)
	eta = (distanceMeasure, distance)
	
	if isinstance(float(sys.argv[7]), float):
			tau = float(sys.argv[7])
	else:
			print("please specify as the 7th argument the tau: [int/float]")
			exit(1)
			
	if isinstance(int(sys.argv[8]),int):
			wbits= int(sys.argv[8])
	else:
			print("please specify as the 8th argument the n of weights bits: [int/float]")
			exit(1)
			
	if isinstance(int(sys.argv[9]),int):
			abits= int(sys.argv[9])
	else:
			print("please specify as the 8th argument the n of activ. bits: [int/float]")
			exit(1)
			
	seed=int(sys.argv[10])
	print ("Seed:"+str(seed))
else:
    print ("Incorrect parameters")
    exit(1)
 
#CudaMemorySettings()
CpuMemorySettings()

nameFile="seed_"+str(seed)+"_"+str(dataSetName)+"_"+str(image_index)+"_Wbits"+str(wbits)+"Abits"+str(abits)+".txt"

# calling algorithms
dc = DataCollection("%s_%s_%s_%s_%s_%s_%s" % (dataSetName, bound, tau, gameType, image_index, eta[0], eta[1]))
dc.initialiseIndex(image_index)

if bound == 'ub':
    (elapsedTime, newConfident, percent, l2dist, l1dist, l0dist, maxFeatures) = (
        upperbound(dataSetName, bound, tau, gameType, image_index, eta, wbits, abits,nameFile,seed))

    dc.addRunningTime(elapsedTime)
    dc.addConfidence(newConfident)
    dc.addManipulationPercentage(percent)
    dc.addl2Distance(l2dist)
    dc.addl1Distance(l1dist)
    dc.addl0Distance(l0dist)
    dc.addMaxFeatures(maxFeatures)

elif bound == 'lb':
    lowerbound(dataSetName, image_index, gameType, eta, tau, wbits, abits)

else:
    print("lower bound algorithm is developing...")
    exit

dc.provideDetails()
dc.summarise()
dc.close()

K.clear_session()
