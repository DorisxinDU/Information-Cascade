# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 16:02:16 2018

@author: xd3y15
"""
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
import keras
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense
import os
import scipy.io
import matplotlib.pyplot as plt
#===========================The functions used==========================================================================
def generateAllBinaryMatrix(n):
  "generate all possible valu of size fixed binary matrix"
  "n is the dimension of the input"
  m=2**n
  G=np.zeros((m,n))
  cordx=list(map(int,np.linspace(0,m-1,m-1)))
  cx=np.array(cordx)
  for j in range(cx.shape[0]):
      G[j,:]=[x for x in bin(j)[2:].zfill(n)]
  return G
def data_shuffle(data_sets_org, percent_of_train, min_test_data=100, shuffle_data=True):
    """Divided the data to train and test and shuffle it"""
    perc = lambda i, t: np.rint((i * t) / 100).astype(np.int32)
    C = type('type_C', (object,), {})
    data_sets = C()
    stop_train_index = perc(percent_of_train[0], data_sets_org.shape[0])
    start_test_index = stop_train_index
    if percent_of_train > min_test_data:
        start_test_index = perc(min_test_data, data_sets_org.shape[0])
    data_sets.train = C()
    data_sets.test = C()
    if shuffle_data:
        shuffled_data = shuffle_in_unison_inplace(data_sets_org)
    else:
        shuffled_data = data_sets_org, data_sets_org.labels
    data_sets.train.data = shuffled_data[:stop_train_index, :]
    data_sets.test.data = shuffled_data[start_test_index:, :]
    return data_sets
def shuffle_in_unison_inplace(a):
    """Shuffle the arrays randomly"""
    p = np.random.permutation(len(a))
    return a[p]
def save_data(Data_set,prediction_classes_of_all,data_saving_path):
    "Save generated dataset"
    C = type('type_C', (object,), {})
    data_sets = C()
    data_sets.data=Data_set
    data_sets.label=prediction_classes_of_all
    scipy.io.savemat(data_saving_path+'dataset.mat', mdict={'data_sets.data': data_sets.data,'data_sets.label':data_sets.label})
    return data_sets
#==============================================================================================================================================
"The functions needed for creating network"
def connectOutputBlock(modelToConnectOut,outputsize):
    "The final prediction block"
    modelToConnectOut.add(Dense(outputsize,activation=activation,name='Prediction_Layer'))
    return modelToConnectOut
def initialblock(structure_nodes,model,X_train,activation,layernum):
    "The initialization setting of the network"
    model.add(Dense(structure_nodes,activation=activation,input_dim=X_train.shape[1],kernel_initializer=keras.initializers.RandomUniform(minval=-5, maxval=5, seed=None),name='Layer'+str(layernum)))
    return model
#========================The defination of the parameters we need==============================================================================
def parameters(i):
      "the structure can alter, e.g:list(map(int,np.linspace(2,100,20))) or [100,50,30,20,15,10]"
      structure=[10,7,5,3]# The structure  of the network (The number of nodes in each layer)
      outputsize=1          # The dimension of the prediction layer
      num_of_experment=str(structure)+'_experment_'+str(i)  #for change the saving name of results
      outputsize=1 # The size of the output
      lr = 0.001  #learning rate
      optimizer=Adam(lr=lr) # The name of the optimizer
      batch_size=100 # The siz of the batch_size
      nb_epoch=300  # The number of training epoch
      activation='tanh'#The activation function
      namecase=num_of_experment+'layer_output_case'
      use_generated_structure=False
      return lr,optimizer,batch_size,nb_epoch,activation,namecase,structure,use_generated_structure,num_of_experment,outputsize
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def creating_network(structure,X_train,activation,outputsize):
    print('Creating MLP network.....')
    for layernum in range(len(structure)):
         if layernum==0:
             model = Sequential()
             model=initialblock(structure[layernum],model,X_train,activation,layernum)
         else:
             model.add(Dense(structure[layernum],activation=activation,name='Layer'+str(layernum)))
    model = connectOutputBlock(model,outputsize)
    print('The structure of MLP network:')
    model.summary()
    return model
#=========================================loop of generating process==================================================================================

        #                          create input
        #===========================================================================================================================
for the_num_of_experment in range(50):  # set the number of running code
        input_dimension=12
        percent_of_train=np.array([65.00])
        Data_set=generateAllBinaryMatrix(input_dimension)
        Data_set_shuffled=data_shuffle(Data_set,percent_of_train,shuffle_data=True)
        X_train,X_test=Data_set_shuffled.train.data,Data_set_shuffled.test.data
        _,_,batch_size,_,activation,namecase,structure,use_generated_structure,num_of_experment,outputsize=parameters(the_num_of_experment)
        #=====================================create network used to generate dataset=======================================================================================
        created_model=creating_network(structure,X_train,activation,outputsize)
        # ===========================saving model================================================================
        model_path='model_save/model_'+num_of_experment # The path of saving model
        if not os.path.exists(model_path):   # if model saving path doesn't exist, create it
        	 os.makedirs(model_path)
#        created_model.save('model/new_model_'+ str(num_of_experment)+'_data_geneator.h5')
        created_model.save(model_path+'/'+'model.h5')
        print('model is generated')
        #=====================================================================================================================================================================
        # ====================================================================================================
        #                                     prediction
        # ===============================Direct result of tanh function================================================================
        prediction_of_all=created_model.predict(Data_set)
        prediction_train=created_model.predict(X_train)
        prediction_test=created_model.predict(X_test)
        # #=================================The output of classification =========================================================
        prediction_classes_of_all=created_model.predict_classes(Data_set)
        prediction_class_train=created_model.predict_classes(X_train)
        prediction_class_test=created_model.predict_classes(X_test)
        del created_model
        data_saving_path='data_saving/'+num_of_experment+'/'
        if not os.path.exists(data_saving_path):   # if model saving path doesn't exist, create it
            	os.makedirs(data_saving_path)
        Dataset_generated=save_data(Data_set,prediction_classes_of_all,data_saving_path)
        # =====================================plot generated complex problem======================================================
        result_path='result/'+num_of_experment+'/'
        if not os.path.exists(result_path):   # if model saving path doesn't exist, create it
            	os.makedirs(result_path)
        fig=plt.figure()
        plt.plot(prediction_of_all,'g.')
        plt.title('output_of_tanh_function_scatter_figure')
        fig.savefig(result_path+'output_of_tanh_function_scatter_figure.png')
        #plt.show()
        fig=plt.figure()
        plt.plot(prediction_train,'b.',label='prediction of train')
        plt.plot(prediction_test,'c.',label='prediction of test')
        plt.legend(loc='lower center')
        plt.title('prediction')
        fig.savefig(result_path+'output_of_prediction_divided'+num_of_experment+'.png')
        #plt.show()
        #--------------------------------------------------------------------------------------
        fig=plt.figure()
        arr=plt.hist(prediction_classes_of_all,label='prediction_classes')
        plt.title('prediction_classes distrbution of all data')
        plt.legend(loc='lower center')
        #plt.show()
        fig.savefig(result_path+'prediction_classes distrbution of all data'+num_of_experment+'.png')
        fig=plt.figure()
        plt.plot(prediction_class_train,'g.',label='prediction classes of train')
        plt.plot(prediction_class_test,'r.',label='prediction classes of test')
        plt.legend(loc='lower center')
        plt.title('prediction')
        fig.savefig(result_path+'output_of_prediction_classes.png')
        #plt.show()
        #==================== plot all in one figure================================================================================
        newpath='result/all shown in one'
        if not os.path.exists(newpath):   # if model saving path doesn't exist, create it
            	os.makedirs(newpath)
        fig_all=plt.figure(figsize=(28, 16))
        fig_all.add_subplot(2,2,1)
        plt.plot(prediction_of_all,'g.')
        plt.title('output_of_tanh_function_scatter_figure')
        fig_all.add_subplot(2,2,2)
        plt.plot(prediction_train,'b.',label='prediction of train')
        plt.plot(prediction_test,'c.',label='prediction of test')
        plt.legend(loc='lower center')
        plt.title('output_of_tanh_function divided')
        fig_all.add_subplot(2,2,3)
        plt.plot(prediction_class_train,'g.',label='prediction classes of train')
        plt.plot(prediction_class_test,'r.',label='prediction classes of test')
        plt.legend(loc='lower center')
        plt.title('prediction of classes')
        fig_all.add_subplot(2,2,4)
        plt.hist(prediction_classes_of_all,label='prediction_classes')
        plt.hist(prediction_class_train,label='prediction_classes train',alpha=0.3)
        plt.hist(prediction_class_test,label='prediction_classes test',alpha=0.3)
        plt.title('prediction_classes distrbution of all data')
        plt.legend(loc='lower center')
        plt.suptitle('all in one'+num_of_experment)
#        plt.show()
        fig_all.savefig(newpath+'/all in one'+num_of_experment+'.png')
    
        