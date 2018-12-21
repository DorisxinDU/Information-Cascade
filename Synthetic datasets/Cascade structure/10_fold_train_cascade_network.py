# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 11:06:05 2018

@author: TOSHIBA
"""
"The process of cascade learning"
#=====================================================================================================
#                           The libraries we need
#=====================================================================================================
#=============reduce the storage==============================
"For avoiding programme using unecessary memory"
import tensorflow as tf
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

#==============================================================
import numpy as np
import scipy.io as sio
from keras.layers.core import Dense
import keras
from keras.models import Sequential
from keras.models import Model
import os
import pickle as cPickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from sklearn.model_selection import KFold 
from training_parameters import parameters 
#================================================================================================
def load_generated_data(data_name):
    "load generated data"
    d=sio.loadmat(data_name)
    return d
def shuffle_in_unison_inplace(a, b):
    """Shuffle the arrays randomly"""
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
def data_label_shuffle(data_sets_org, percent_of_train, min_test_data=80, shuffle_data=True):
    """Divided the data to train and test and shuffle it"""
    perc = lambda i, t: np.rint((i * t) / 100).astype(np.int32)
    C = type('type_C', (object,), {})
    data_sets = C()
    stop_train_index = perc(percent_of_train[0], data_sets_org['data_sets.data'].shape[0])
    start_test_index = stop_train_index
    if percent_of_train > min_test_data:
        start_test_index = perc(min_test_data, data_sets_org['data_sets.data'].shape[0])
    data_sets.train = C()
    data_sets.test = C()
    if shuffle_data:
        shuffled_data, shuffled_labels = shuffle_in_unison_inplace(data_sets_org['data_sets.data'], data_sets_org['data_sets.label'])
    else:
        shuffled_data, shuffled_labels = data_sets_org.data, data_sets_org.labels
    data_sets.train.data = shuffled_data[:stop_train_index, :]
    data_sets.train.labels = shuffled_labels[:stop_train_index, :]
    data_sets.test.data = shuffled_data[start_test_index:, :]
    data_sets.test.labels = shuffled_labels[start_test_index:, :]
    return data_sets
def store_file(stringOfHistory,his):
    "Store the history of accuarcy"
    with open(stringOfHistory,'wb') as fp:
        return cPickle.dump(his,fp)
def load_DATA(name):
    "Load history"
    with open(name, 'rb') as f:
        data=cPickle.load(f)
        return data
#==============================================================================================================================================
"The functions needed for creating network"
def connectOutputBlock(modelToConnectOut,outputsize,activation):
    "The final prediction block"
    modelToConnectOut.add(Dense(outputsize,activation=activation,name='Prediction_Layer'))
    return modelToConnectOut
def initialblock(structure_nodes,model,X_train,activation,layernum):
    "The initialization setting of the network"
    model.add(Dense(structure_nodes,activation=activation,input_dim=X_train.shape[1],kernel_initializer=keras.initializers.RandomUniform(minval=-5, maxval=5, seed=None),name='Layer'+str(layernum)))
    return model
def trainModel(modelToTrain,data,currentEpochs,batch_size,optimizer,accuarcy_name,filepath):
  modelToTrain.compile(loss='mean_squared_error',optimizer=optimizer,metrics=[accuarcy_name])
  trainingResults = modelToTrain.fit(data[0],data[1], batch_size=batch_size, nb_epoch=currentEpochs, validation_data=(data[2],data[3]),verbose=1,callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=(currentEpochs/10), verbose=0, mode='auto', baseline=None)])
  hist = trainingResults.history
  return hist
#=====================================================================================================================
#                load data and shuffled
#====================================================================
for number_of_data in [17,28,44,48]:
    for num_of_train in range(1):      
        lr,optimizer,batch_size,nb_epoch,activation,structure,num_of_experment,outputsize,percent_of_train,accuarcy_name,name_num_of_experment=parameters(number_of_data,num_of_train)
        Data_set=load_generated_data('data_saving/'+name_num_of_experment+'/dataset.mat')
        Data,label=Data_set['data_sets.data'],Data_set['data_sets.label']
        num_of_experment1=num_of_experment
        # =========================================================================================================
        kf = KFold(n_splits=10)
        fold=0
        kf.get_n_splits(Data) # returns the number of splitting iterations in the cross-validator
        for train_index, test_index in kf.split(Data):
            print('TRAIN:', train_index[-8:-1], 'TEST:', test_index[-8:-1])
            X_train, X_test = Data[train_index],Data[test_index]
            Y_train, Y_test = label[train_index], label[test_index]
            fold=fold+1
            num_of_experment='fold_'+str(fold)+num_of_experment1
        #===========================================================================================
        #                                        Cascade Training process
        #===========================================================================================
            print('TRAINING Cascade learning')
            history = dict()
            new_data_set=[X_train, Y_train,X_test, Y_test]
            #==============================training process=======================================================
            for layernum in range(len(structure)):
                  if layernum==0:
                      model = Sequential()
                      model=initialblock(structure[layernum],model,X_train,activation,layernum)
                      model = connectOutputBlock(model,outputsize,activation)   
                      model.summary()
                      filepath='check_point_model_layer_save/model_'+num_of_experment+'/layer_'+str(layernum) # The path of saving model
                      if not os.path.exists(filepath):
                            os.makedirs(filepath)
                      history['iter'+str(layernum)] = []
                      history['iter'+str(layernum)].append(trainModel(model,new_data_set,nb_epoch,batch_size,optimizer,accuarcy_name,filepath+"/weights-improvement-{epoch:02d}.hdf5"))
                      X_train1,X_test1=X_train,X_test
                      model_layer_path='model_layer_save/model_'+num_of_experment # The path of saving model
                      if not os.path.exists(model_layer_path):
                          os.makedirs(model_layer_path)
                      model.save(model_layer_path+'/'+str(layernum)+'model.h5')
                      print('layer_'+str(layernum)+'_model is generated')
                  else:
                      intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer('Layer'+str(layernum-1)).output)
                      prediction_class_train_updata=intermediate_layer_model.predict(X_train1)
                      prediction_class_test_updata=intermediate_layer_model.predict(X_test1)
                      new_data_set=[prediction_class_train_updata, Y_train,prediction_class_test_updata, Y_test] 
                      del model,intermediate_layer_model
                      model= Sequential()
                      model=initialblock(structure[layernum],model,prediction_class_train_updata,activation,layernum)
                      model = connectOutputBlock(model,outputsize,activation)  
                      model.summary()
                      filepath='check_point_model_layer_save/model_'+num_of_experment+'/layer_'+str(layernum) # The path of saving model
                      if not os.path.exists(filepath):
                        os.makedirs(filepath)
                      history['iter'+str(layernum)] = []
                      history['iter'+str(layernum)].append(trainModel(model,new_data_set,nb_epoch*(layernum+1),batch_size,optimizer,accuarcy_name,filepath+"/weights-improvement-{epoch:02d}.hdf5"))
                      X_train1,X_test1=new_data_set[0],new_data_set[2]
                      model_layer_path='model_layer_save/model_'+num_of_experment # The path of saving model
                      if not os.path.exists(model_layer_path):
                        os.makedirs(model_layer_path)
                      model.save(model_layer_path+'/'+str(layernum)+'model.h5')
                      print('layer_'+str(layernum)+'_model is generated')
                  
            #==============================plot training process=======================================================
            accuarcy_save_path='history_accuarcy/acc_'+num_of_experment
            if not os.path.exists(accuarcy_save_path):
                os.makedirs(accuarcy_save_path)
            store_file(accuarcy_save_path+'/'+'acc.pkl',history)
            print('history is saved')
            #============================================================================================
            HISTORY=load_DATA(accuarcy_save_path+'/'+'acc.pkl')
            hISTORY=dict()
            for layernum in range(len(structure)):
                hISTORY['iter_TRAIN'+str(layernum)]=[]
                hISTORY['iter_TEST'+str(layernum)]=[]
                for i in range(len(HISTORY['iter'+str(layernum)])):
                      hISTORY['iter_TRAIN'+str(layernum)].append(HISTORY['iter'+str(layernum)][i][accuarcy_name]) 
                      hISTORY['iter_TEST'+str(layernum)].append(HISTORY['iter'+str(layernum)][i]['val_'+accuarcy_name])
            #--------------------------------------------------------------------------------------------------------------
            accuarcy_image_save_path='image_history_accuarcy/acc_'+num_of_experment
            if not os.path.exists(accuarcy_image_save_path):
                os.makedirs(accuarcy_image_save_path)
            fig_all=plt.figure(figsize=(15, 15))
            axis_font=16
            axes=fig_all.add_subplot(1,2,1)
            for layernum in range(len(structure)):
                axes.plot(hISTORY['iter_TRAIN'+str(layernum)][0],label='CAS_train_acc_layer'+str(layernum))
                axes.text(0.5*len(hISTORY['iter_TRAIN'+str(layernum)][0]),0.5+0.1*layernum,'Layer_'+str(layernum)+'_accuarcy is:'+str(hISTORY['iter_TRAIN'+str(layernum)][0][-1]))
            plt.legend()
            plt.grid(True)
            axes.set_xlim(-0.1)
            axes.set_ylim(-0.1)
            axes.set_yticks(np.linspace(0,1,11)) 
            plt.title('ACC_TRAIN',fontsize=axis_font+2)
            plt.xlabel('Epoch',fontsize=axis_font)
            plt.ylabel('ACC',fontsize=axis_font)
            axes=fig_all.add_subplot(1,2,2) 
            for layernum in range(len(structure)):
                axes.plot(hISTORY['iter_TEST'+str(layernum)][0],label='CAS_test_acc_layer'+str(layernum))
                axes.text(0.5*len(hISTORY['iter_TRAIN'+str(layernum)][0]),0.5+0.1*layernum,'Layer_'+str(layernum)+'_accuarcy is:'+str(hISTORY['iter_TEST'+str(layernum)][0][-1]))
            plt.legend()
            plt.grid(True)
            axes.set_xlim(-0.1)
            axes.set_ylim(-0.1)
            axes.set_yticks(np.linspace(0,1,11))
            plt.title('ACC_TEST',fontsize=axis_font+2)
            plt.xlabel('Epoch',fontsize=axis_font)
            plt.ylabel('ACC',fontsize=axis_font)
            plt.suptitle('all in one'+num_of_experment)
            #plt.show()
            fig_all.savefig(accuarcy_image_save_path+'/accuarcy.png')
            fig_all.savefig('image_history_accuarcy/accuarcy'+num_of_experment+'.png')