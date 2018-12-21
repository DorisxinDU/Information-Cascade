# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 11:36:08 2018

@author: xd3y15
"""
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
from keras.optimizers import Adam
from keras.models import Sequential
import os
import pickle as cPickle
import matplotlib.pyplot as plt
from keras.models import Model
from sklearn.model_selection import KFold 
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
# _____________________________________________________________________________________________________________

#==============================================================================================================================================
"The functions needed for creating network"
def connectOutputBlock(modelToConnectOut,outputsize,activation):
    "The final prediction block"
    modelToConnectOut.add(Dense(outputsize,activation=activation,name='Prediction_Layer'))
    return modelToConnectOut
def initialblock(structure_nodes,model,X_train,activation,layernum):
    "The initialization setting of the network"
    model.add(Dense(structure_nodes,activation=activation,input_dim=X_train.shape[1],kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.3, seed=None),name='Layer'+str(layernum)))
    return model
def trainModel(modelToTrain,data,currentEpochs,batch_size,optimizer,accuarcy_name,filepath,structure):
  modelToTrain.compile(loss='mean_squared_error',optimizer=optimizer,metrics=[accuarcy_name])
  trainingResults = modelToTrain.fit(data[0],data[1], batch_size=batch_size, nb_epoch=currentEpochs, validation_data=(data[2],data[3]),verbose=1,callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=currentEpochs/10, verbose=1, mode='auto'),keras.callbacks.ModelCheckpoint(filepath+"/weights-improvement-{epoch:02d}.hdf5", monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)])
  hist = trainingResults.history
  return hist
#========================The defination of the parameters we need==============================================================================
def parameters(number_of_data,num_of_train):
      "the structure can alter, e.g:list(map(int,np.linspace(2,100,20))) or [100,50,30,20,15,10]"
      input_dimension=12
      percent_of_train=np.array([80.00])
      structure=[10,7,5,3]# The structure  of the network (The number of nodes in each layer)
      outputsize=1          # The dimension of the prediction layer
      num_of_experment='test_version_'+str(percent_of_train)+'_'+str(structure)+'_full_size_experment_'+str(number_of_data)+'_'+str(num_of_train)  #for change the saving name of results
      name_num_of_experment=str(structure)+'_experment_'+str(number_of_data)
      outputsize=1 # The size of the output
      lr = 0.001  #learning rate
      optimizer=Adam(lr=lr) # The name of the optimizer
      batch_size=256 # The siz of the batch_size
      nb_epoch=30000 # The number of training epoch
      activation='tanh'#The activation function
      namecase=num_of_experment+'layer_output_case'
      accuarcy_name='acc'
      use_generated_structure=False
      return lr,optimizer,batch_size,nb_epoch,activation,namecase,structure,use_generated_structure,num_of_experment,outputsize,input_dimension,percent_of_train,accuarcy_name,name_num_of_experment
#=====================================================================================================================
#                load data and shuffled
#====================================================================
for number_of_data in [17,28,44,48]:
    for num_of_train in range(1):
        lr,optimizer,batch_size,nb_epoch,activation,namecase,structure,use_generated_structure,num_of_experment,outputsize,input_dimension,percent_of_train,accuarcy_name,name_num_of_experment=parameters(number_of_data,num_of_train)
        Data_set=load_generated_data('data_saving/'+name_num_of_experment+'/dataset.mat')
        Data,label=Data_set['data_sets.data'],Data_set['data_sets.label']
        # ========================================10 fold training===============================================================
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
        # ====================================================================================================
        # Data_set_shuffled=data_label_shuffle(Data_set, percent_of_train,shuffle_data=True)
        # X_train, Y_train,X_test, Y_test=Data_set_shuffled.train.data,Data_set_shuffled.train.labels,Data_set_shuffled.test.data,Data_set_shuffled.test.labels
        # =======================================================================================================
            Dataset=[X_train, Y_train,X_test, Y_test]
            #===========================================================================================
            #                                        Fully_connect Training process
            #===========================================================================================
            print('TRAINING fully_connected learning')
            history = dict()
            output_history=dict()
            # ==================================================================================
            #==============================construct fully  connect network=======================================================
            for layernum in range(len(structure)):
                  if layernum==0:
                      model = Sequential()
                      model=initialblock(structure[layernum],model,X_train,activation,layernum)
                  else:
                      model.add(Dense(structure[layernum],activation=activation,name='Layer'+str(layernum)))
            model=connectOutputBlock(model,outputsize,activation)
            model.summary()
            # ++++++++++++++++++++++++++++++++++++++++++++aining model++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            filepath='check_point_model_layer_save/model_'+num_of_experment
            # The path of saving model_each_epoch
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            history['accuarcy'] = []
            output_history['output'] = []
            hist=trainModel(model,Dataset,nb_epoch,batch_size,optimizer,accuarcy_name,filepath,structure)
            history['accuarcy'].append(hist)
            model_save_path='fully_model_saving/model_'+num_of_experment # The path of saving model
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            model.save(model_save_path+'/model.h5')
            print('fully_model is generated')
            #==============================plot training process=======================================================
            accuarcy_save_path='history_accuarcy/acc_'+num_of_experment
            if not os.path.exists(accuarcy_save_path):
                os.makedirs(accuarcy_save_path)
            store_file(accuarcy_save_path+'/'+'acc.pkl',history)
            print('history is saved')
            #============================================================================================
            HISTORY=load_DATA(accuarcy_save_path+'/'+'acc.pkl')
            hISTORY=dict()
            hISTORY['iter_TRAIN']=[]
            hISTORY['iter_TEST']=[]
            for i in range(len(HISTORY['accuarcy'])):
                  hISTORY['iter_TRAIN'].append(HISTORY['accuarcy'][i][accuarcy_name])
                  hISTORY['iter_TEST'].append(HISTORY['accuarcy'][i]['val_'+accuarcy_name])
            #--------------------------------------------------------------------------------------------------------------
            accuarcy_image_save_path='image_history_accuarcy/acc_'+num_of_experment
            if not os.path.exists(accuarcy_image_save_path):
                os.makedirs(accuarcy_image_save_path)
            fig_all=plt.figure(figsize=(15, 15))
            axis_font=16
            axes=fig_all.add_subplot(1,2,1)
            axes.plot(hISTORY['iter_TRAIN'][0][:],label='train_acc')
            axes.text(0.5*len(hISTORY['iter_TRAIN'][0]),0.5,'The final accuarcy is:'+str(hISTORY['iter_TRAIN'][0][-1]))
            plt.grid(True)
            plt.legend()
            axes.set_xlim(-0.1)
            axes.set_ylim(-0.1)
            axes.set_yticks(np.linspace(0,1,11))
            plt.title('ACC_TRAIN',fontsize=axis_font + 2)
            plt.xlabel('Epoch',fontsize=axis_font)
            plt.ylabel('ACC',fontsize=axis_font)
            axes=fig_all.add_subplot(1,2,2)
            axes.plot(hISTORY['iter_TEST'][0][:],label='test_acc')
            axes.text(0.5*len(hISTORY['iter_TRAIN'][0]),0.5,'The final accuarcy is:'+str(hISTORY['iter_TEST'][0][-1]))
            plt.grid(True)
            plt.legend()
            axes.set_xlim(-0.1)
            axes.set_ylim(-0.1)
            axes.set_yticks(np.linspace(0,1,11))
            plt.title('ACC_TEST',fontsize=axis_font + 2)
            plt.xlabel('Epoch',fontsize=axis_font)
            plt.ylabel('ACC',fontsize=axis_font)
            plt.suptitle('all in one'+num_of_experment)
            #plt.show()
            fig_all.savefig(accuarcy_image_save_path+'/accuarcy.png')
            fig_all.savefig('image_history_accuarcy/accuarcy'+num_of_experment+'.png')

