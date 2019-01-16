import tensorflow as tf
from keras.models import Sequential, Model
from keras.backend.tensorflow_backend import set_session
from models.model_factory import build_model_QNN
from utils.config_utils import Config
from utils.load_data import load_dataset
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras.optimizers import SGD, Adam, Adadelta
from keras.losses import squared_hinge
import keras.backend as K
import numpy
import os

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


def getModelFromQNN(cf,train_x,train_y,test_x,test_y):
    print('Construct the Network\n')
    model = build_model_QNN(cf)
    print('setting up the network and creating callbacks\n')
    early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=10, mode='min', verbose=1)
    checkpoint = ModelCheckpoint(cf.out_wght_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)
    tensorboard = TensorBoard(log_dir='./logs/' + str(cf.tensorboard_name), histogram_freq=0, write_graph=True, write_images=False)
    print('loading data\n')

    # learning rate schedule
    def scheduler(epoch):
        if epoch in cf.decay_at_epoch:
            index = cf.decay_at_epoch.index(epoch)
            factor = cf.factor_at_epoch[index]
            lr = K.get_value(model.optimizer.lr)
            IT = train_x.shape[0]/cf.batch_size
            current_lr = lr * (1./(1.+cf.decay*epoch*IT))
            K.set_value(model.optimizer.lr,current_lr*factor)
            print('\nEpoch {} updates LR: LR = LR * {} = {}\n'.format(epoch+1,factor, K.get_value(model.optimizer.lr)))
        return K.get_value(model.optimizer.lr)

    lr_decay = LearningRateScheduler(scheduler)
    #sgd = SGD(lr=cf.lr, decay=cf.decay, momentum=0.9, nesterov=True)
    adam= Adam(lr=cf.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=cf.decay)
    try:
      print("Weights path: "+str(cf.out_wght_path))
      model.load_weights(cf.out_wght_path)
      print('Load previous weights\n')
      model.compile(loss=squared_hinge, optimizer=adam, metrics=['accuracy'])
	#CpuMemorySettings()
      return False, model
	#Trained? NO weights exist
    except  e:
	#CudaMemorySettings()
      print('Failed:'+ str(e))
    print('compiling the network\n')
    model.compile(loss=squared_hinge, optimizer=adam, metrics=['accuracy'])
    print('No weights preloaded, training from scratch\n')
    
    model.fit(train_x,train_y,
        batch_size = cf.batch_size,
        epochs = cf.epochs,
        verbose = cf.progress_logging,
        callbacks = [checkpoint, tensorboard,lr_decay],
        validation_data = (test_x,test_y))
    #Trained? YES weights so not exist
    
    model.load_weights(cf.out_wght_path)
    #model.compile(loss=squared_hinge, optimizer=adam, metrics=['accuracy'])

    return True, model
    
def getModelFromDeepGame(cf, x_train,y_train,x_test,y_test,epochs,batch_size):
    print('Construct the Network\n')
    model = build_model(cf,"Deep")
    print('setting up the network and creating callbacks\n')

    checkpoint = ModelCheckpoint(cf.out_wght_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adadelta(),
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              callbacks = [checkpoint],
              shuffle=True,
              validation_data=(x_test, y_test))
    
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    return model

def loadDataset(cf):
    if cf.dataset=="MNIST":
        return load_dataset(cf.dataset,50000,55000)
    else:
        return load_dataset(cf.dataset,45000,50000)
