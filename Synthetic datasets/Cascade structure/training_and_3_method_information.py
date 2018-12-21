# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 10:19:43 2018

@author: TOSHIBA
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 11:06:05 2018

@author: TOSHIBA
"""
"The process of training cascade learning and calculate mutual information"
#=====================================================================================================
#                           The libraries we need
#=====================================================================================================
#=============Reduce the storage used==============================
"For avoiding programme using unecessary memory"
import tensorflow as tf
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
#======================Import the libraries========================================
import numpy as np
import scipy.io as sio
from keras.layers.core import Dense
import keras
from keras.optimizers import Adam
from keras.models import Sequential
from keras.models import Model
import tqdm
import time
import keras.backend as K
from time import sleep
import os
import pickle as cPickle
import matplotlib.pyplot as plt
from Cascade_calculate_loaded_data_information1 import calculate_layer_mutual_information,extract_probs,calculate_mutual_information,calculate_lower_mutual_information # Functions of calculating mutual information by different methods
from training_parameters import parameters # Import parametrs used for training model
#==========================The functions we defined======================================================================
def load_generated_data(data_name):
    "load generated data"
    d=sio.loadmat(data_name)
    return d
def shuffle_in_unison_inplace(a, b):
    "Shuffle the dataset randomly"
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
def data_label_shuffle(data_sets_org, percent_of_train, min_test_data=80, shuffle_data=True):
    "Split the data to train and test"
    perc = lambda i, t: np.rint((i * t) / 100).astype(np.int32) # Calculate the index of train and test dataset according to percentage 
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
    "Load saved history"
    with open(name, 'rb') as f:
        data=cPickle.load(f)
        return data
#========================================================================================================================================================================================
class OutputObserver(keras.callbacks.Callback):
    'callback to observe the output of the network'
    def __init__(self, data_set,layernum,DATA,Label,bins,logs={}):
        self.out_log = dict()  # Save the output of each layer in each epoch
        self.out_log['eachlayer']=[]
        self.out_log['each_final_layer']=[]
        self.data_set = data_set
        self.layer_name='Layer'+str(layernum)
        self.DATA=DATA 
        self.Label=Label
        self.bins=bins
        self.local_IXT_cas_layer,self.local_ITY_cas_layer,self.finallocal_IXT_cas_layer,self.finallocal_ITY_cas_layer=[],[],[],[] # Save the information calculated by Binning method
        self.KDElocal_IXT_cas_layer,self.KDElocal_ITY_cas_layer,self.KDEfinallocal_IXT_cas_layer,self.KDEfinallocal_ITY_cas_layer=[],[],[],[] # Save the information calculated by PWD based upper bound method
        self.kdislocal_IXT_cas_layer,self.kdislocal_ITY_cas_layer,self.kdisfinallocal_IXT_cas_layer,self.kdisfinallocal_ITY_cas_layer=[],[],[],[]  # Save the information calculated by PWD based lower bound method
        self.pys1,self.unique_inverse_x,self. unique_inverse_y,self. pxs = extract_probs(self.Label,self.data_set) 
        self.PXs, self.PYs = np.asarray(self.pxs).T, np.asarray(self.pys1).T  # The distributation of Input and Label 
    def on_epoch_end(self, epoch, logs={}):
        "Calculate and save the information for each epoch"
        intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(self.layer_name).output) 
        layeroutput_cas=intermediate_layer_model.predict(self.data_set)                                           # Get the output of layer with name self.layer_name
        intermediate_predict_layer_model = Model(inputs=model.input,outputs=model.get_layer('Prediction_Layer').output) 
        finallayeroutput_cas=intermediate_predict_layer_model.predict(self.data_set)                              # Get the output of the classifier which is following the layer with name self.layer_name
        self.out_log['eachlayer'].append(layeroutput_cas)
        self.out_log['each_final_layer'].append(finallayeroutput_cas)
#        ==============================================================
        IX_CAS2,IY_CAS2=calculate_lower_mutual_information(layeroutput_cas[0:len(self.DATA)],self.PXs, self.PYs,self.unique_inverse_x,self.unique_inverse_y) # Calculate information for cascaded layer by PWD based lower bound method
        finalIX_CAS2,finalIY_CAS2=calculate_lower_mutual_information(finallayeroutput_cas[0:len(self.DATA)],self.PXs, self.PYs,self.unique_inverse_x,self.unique_inverse_y) # Calculate information for the classifier following the cascaded layer
        print('KdisI(X;T) is:'+str(IX_CAS2),'KdisI(Y;T) is:'+str(IY_CAS2))
        print('KdisI(X;T) is:'+str(finalIX_CAS2),'KdisI(Y;T) is:'+str(finalIY_CAS2))
        self.kdislocal_IXT_cas_layer.append(IX_CAS2)
        self.kdislocal_ITY_cas_layer.append(IY_CAS2)
        self.kdisfinallocal_IXT_cas_layer.append(finalIX_CAS2)
        self.kdisfinallocal_ITY_cas_layer.append(finalIY_CAS2)
#        ==============================================================
        IX_CAS,IY_CAS=calculate_mutual_information(layeroutput_cas[0:len(self.DATA)],self.PXs, self.PYs,self.unique_inverse_x,self.unique_inverse_y) # Calculate information for cascaded layer  by PWD based upper bound method
        finalIX_CAS,finalIY_CAS=calculate_mutual_information(finallayeroutput_cas[0:len(self.DATA)],self.PXs, self.PYs,self.unique_inverse_x,self.unique_inverse_y) # Calculate information for the classifier following the cascaded layer  by PWD based upper bound method
        print('KDEI(X;T) is:',IX_CAS,'KDEI(Y;T) is:',IY_CAS)
        print('KDEI(X;T) is:',finalIX_CAS,'KDEI(Y;T) is:',finalIY_CAS)
        self.KDElocal_IXT_cas_layer.append(IX_CAS)
        self.KDElocal_ITY_cas_layer.append(IY_CAS)
        self.KDEfinallocal_IXT_cas_layer.append(finalIX_CAS)
        self.KDEfinallocal_ITY_cas_layer.append(finalIY_CAS)
        
#        ================================================================================
        IX_CAS1,IY_CAS1=calculate_layer_mutual_information(layeroutput_cas[0:len(self.DATA)],self.PXs, self.PYs,self.unique_inverse_x,self.unique_inverse_y,self.bins) # Calculate information for cascaded layer by Binning method
        finalIX_CAS1,finalIY_CAS1=calculate_layer_mutual_information(finallayeroutput_cas[0:len(self.DATA)],self.PXs, self.PYs,self.unique_inverse_x,self.unique_inverse_y,self.bins) # Calculate information for the classifier following the cascaded layer by Binning method
        self.local_IXT_cas_layer.append(IX_CAS1)
        self.local_ITY_cas_layer.append(IY_CAS1)
        self.finallocal_IXT_cas_layer.append(finalIX_CAS1)
        self.finallocal_ITY_cas_layer.append(finalIY_CAS1)
#        ==============================================================
"The functions needed for creating network"
def connectOutputBlock(modelToConnectOut,outputsize,activation):
    "The final prediction block"
    modelToConnectOut.add(Dense(outputsize,activation=activation,name='Prediction_Layer'))
    return modelToConnectOut
def initialblock(structure_nodes,model,X_train,activation,layernum):
    "The initialization setting of the network"
    model.add(Dense(structure_nodes,activation=activation,input_dim=X_train.shape[1],kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01),kernel_initializer=keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None),name='Layer'+str(layernum)))
    return model

def trainModel(modelToTrain,data,currentEpochs,batch_size,optimizer,accuarcy_name,filepath,layernum,DATA,Label):
    "The way of training model and save the relative results"
    modelToTrain.compile(loss='mean_squared_error',optimizer=optimizer,metrics=[accuarcy_name])
    bins = np.linspace(-1, 1, 25) # The intervals used in Binning method 
    bins = bins.astype(np.float32) 
    prediction_logs=OutputObserver(np.concatenate((data[0],data[2])),layernum,DATA,np.concatenate((data[1],data[3])),bins) # The function used to calculate mutual information for each epoch
    trainingResults = modelToTrain.fit(data[0],data[1], batch_size=batch_size, nb_epoch=currentEpochs, validation_data=(data[2],data[3]),verbose=1,callbacks=[keras.callbacks.ModelCheckpoint(filepath+"/weights-improvement-{epoch:02d}.hdf5", monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1),prediction_logs])
    hist = trainingResults.history 
    # If you want to check the result by early stooping, add 'keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=nb_epoch/10, verbose=1, mode='auto')' into callbacks
    return hist,prediction_logs.out_log,prediction_logs.local_IXT_cas_layer,prediction_logs.local_ITY_cas_layer,prediction_logs.finallocal_IXT_cas_layer,prediction_logs.finallocal_ITY_cas_layer,prediction_logs
#==================================================================The defination of the parameters we need==============================================================================

#=====================================================================================================================
#                                          Start training process
#=====================================================================================================================

for number_of_data in [17]: # The number of generated datasets you are going to use, [17,28,44,48] are the datasets we used in paper
    for num_of_train in range(1): # The number of training times for get average information plane over multi times training
        lr,optimizer,batch_size,nb_epoch,activation,structure,num_of_experment,outputsize,percent_of_train,accuarcy_name,name_num_of_experment=parameters(number_of_data,num_of_train) # Load the parameters needed
        Data_set=load_generated_data('data_saving/'+name_num_of_experment+'/dataset.mat') # Load datasets and label
        Data,Label=Data_set['data_sets.data'],Data_set['data_sets.label'] 
        DATA=Data # Rename data for further using and avoid covering
        Data_set_shuffled=data_label_shuffle(Data_set, percent_of_train,shuffle_data=True)
        X_train, Y_train,X_test, Y_test=Data_set_shuffled.train.data,Data_set_shuffled.train.labels,Data_set_shuffled.test.data,Data_set_shuffled.test.labels # Split train and test datasets
        #=============================================================================================================
        #                                        Cascade Training process
        #=============================================================================================================

#        ---------------------------------------------------
        print('TRAINING Cascade learning')
        history = dict() # Save the performance of each epoch
        output_history=dict() # Save the out put of each epoch and layer
        new_data_set=[X_train, Y_train,X_test, Y_test]
        # ==================================================================================
        sleep(0.1)
        #====================================================================training process==========================================================
        local_IXT_cas,local_ITY_cas,finallocal_IXT_cas, finallocal_ITY_cas=[],[],[],[] # Save information for each layer by Binning method
        KDElocal_IXT_cas, KDElocal_ITY_cas,KDEfinallocal_IXT_cas, KDEfinallocal_ITY_cas=[],[],[],[] # Save information for each layer by PWD upper bound method
        Kdislocal_IXT_cas, Kdislocal_ITY_cas,Kdisfinallocal_IXT_cas, Kdisfinallocal_ITY_cas=[],[],[],[] # Save information for each layer by PWD lower bound method
        for layernum in tqdm.tqdm(range(len(structure))): # Go through each layer cascaded, and tqdm used for showing progress bar 
              if layernum==0: # For the first layer, the input is the original datasets 
                  model = Sequential()
                  model=initialblock(structure[layernum],model,X_train,activation,layernum)
                  model = connectOutputBlock(model,outputsize,activation)
                  model.summary()
                  filepath='check_point_model_layer_save/model_'+num_of_experment+'/layer_'+str(layernum) # The path of saving model
                  if not os.path.exists(filepath):
                        os.makedirs(filepath)
                  history['iter'+str(layernum)] = []
                  output_history['iter'+str(layernum)] = []
                  hist,prediction_logs_out,local_IXT_cas_layer, local_ITY_cas_layer,finallocal_IXT_cas_layer, finallocal_ITY_cas_layer,prediction_logs=trainModel(model,new_data_set,nb_epoch,batch_size,optimizer,accuarcy_name,filepath,layernum,DATA,Label) # Get trained parameters and corresponding information 
                  history['iter'+str(layernum)].append(hist)
                  output_history['iter'+str(layernum)].append(prediction_logs_out)
                  X_train1,X_test1=X_train,X_test                                 # Copy the original data and label for avoiding cover and further using  
                  model_layer_path='model_layer_save/model_'+num_of_experment     # The path of saving model
                  if not os.path.exists(model_layer_path):                        # Create the file for saving model if this file is not existing
                      os.makedirs(model_layer_path)
                  model.save(model_layer_path+'/'+str(layernum)+'model.h5')
                  print('layer_'+str(layernum)+'_model is generated')
                  # =======================================================================================================================
                  #                        Save all mutual information given by different methods for first layer 
                  # =======================================================================================================================
                  local_IXT_cas.append(local_IXT_cas_layer)                       
                  local_ITY_cas.append(local_ITY_cas_layer)
                  finallocal_IXT_cas.append(finallocal_IXT_cas_layer)
                  finallocal_ITY_cas.append(finallocal_ITY_cas_layer)
                  # ========================================
                  KDElocal_IXT_cas.append(prediction_logs.KDElocal_IXT_cas_layer)
                  KDElocal_ITY_cas.append(prediction_logs.KDElocal_ITY_cas_layer)
                  KDEfinallocal_IXT_cas.append(prediction_logs.KDEfinallocal_IXT_cas_layer)
                  KDEfinallocal_ITY_cas.append(prediction_logs.KDEfinallocal_ITY_cas_layer)
                  # =============================================
                  Kdislocal_IXT_cas.append(prediction_logs.kdislocal_IXT_cas_layer)
                  Kdislocal_ITY_cas.append(prediction_logs.kdislocal_ITY_cas_layer)
                  Kdisfinallocal_IXT_cas.append(prediction_logs.kdisfinallocal_IXT_cas_layer)
                  Kdisfinallocal_ITY_cas.append(prediction_logs.kdisfinallocal_ITY_cas_layer)
                  # ===============================================================================================================================
              else: # Except first layer, the input is the output of previous layer
                  intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer('Layer'+str(layernum-1)).output) # Get the output of previous hidden layer, namely get the new input of the cascaded layer
                  prediction_class_train_updata=intermediate_layer_model.predict(X_train1) # Training part
                  prediction_class_test_updata=intermediate_layer_model.predict(X_test1)   # Testing part
                  new_data_set=[prediction_class_train_updata, Y_train,prediction_class_test_updata, Y_test] # Constructed dataset and label together, for convenient training
                  del model,intermediate_layer_model                                       # Delete previous model and constructed now cascaded model
                  model= Sequential()                                                      # Define the new model
                  model=initialblock(structure[layernum],model,prediction_class_train_updata,activation,layernum)
                  model = connectOutputBlock(model,outputsize,activation)
                  model.summary()
                  filepath='check_point_model_layer_save/model_'+num_of_experment+'/layer_'+str(layernum)                 # The path of saving model
                  if not os.path.exists(filepath):                                                                        # If the path is not existing, create it
                        os.makedirs(filepath)
                  history['iter'+str(layernum)] = []
                  output_history['iter'+str(layernum)] = []
                  hist,prediction_logs_out,local_IXT_cas_layer, local_ITY_cas_layer,finallocal_IXT_cas_layer,finallocal_ITY_cas_layer,prediction_logs=trainModel(model,new_data_set,nb_epoch,batch_size,optimizer,accuarcy_name,filepath,layernum,DATA,Label)#The training epoch can changed into nb_epoch*(layernum+1) as paper said
                  history['iter'+str(layernum)].append(hist)
                  output_history['iter'+str(layernum)].append(prediction_logs_out)
                  X_train1,X_test1=new_data_set[0],new_data_set[2]            # Copy the input of current cascaded layer for further using
                  model_layer_path='model_layer_save/model_'+num_of_experment # The path of saving model
                  if not os.path.exists(model_layer_path):                    # If the path is not existing, create it
                    os.makedirs(model_layer_path)
                  model.save(model_layer_path+'/'+str(layernum)+'model.h5')
                  print('layer_'+str(layernum)+'_model is generated')
                  # =======================================================================================================================
                  #                        Save all mutual information given by different methods for remaining layers
                  # =======================================================================================================================
#                  =========================================================
                  local_IXT_cas.append(local_IXT_cas_layer)
                  local_ITY_cas.append(local_ITY_cas_layer)
                  finallocal_IXT_cas.append(finallocal_IXT_cas_layer)
                  finallocal_ITY_cas.append(finallocal_ITY_cas_layer)
#                  ===============================================================
                  KDElocal_IXT_cas.append(prediction_logs.KDElocal_IXT_cas_layer)
                  KDElocal_ITY_cas.append(prediction_logs.KDElocal_ITY_cas_layer)
                  KDEfinallocal_IXT_cas.append(prediction_logs.KDEfinallocal_IXT_cas_layer)
                  KDEfinallocal_ITY_cas.append(prediction_logs.KDEfinallocal_ITY_cas_layer)
#                  =============================================
                  Kdislocal_IXT_cas.append(prediction_logs.kdislocal_IXT_cas_layer)
                  Kdislocal_ITY_cas.append(prediction_logs.kdislocal_ITY_cas_layer)
                  Kdisfinallocal_IXT_cas.append(prediction_logs.kdisfinallocal_IXT_cas_layer)
                  Kdisfinallocal_ITY_cas.append(prediction_logs.kdisfinallocal_ITY_cas_layer)
#                 ===============================================================================================================================
        "Save all the mutual information as files"
        information_save_path='cascade_information_save/'+num_of_experment  # The path of saving mutual information
        if not os.path.exists(information_save_path):
            os.makedirs(information_save_path)
        store_file(information_save_path+'/information_x.pkl',local_IXT_cas)
        store_file(information_save_path+'/information_y.pkl',local_ITY_cas)
        store_file(information_save_path+'/final_information_x.pkl',finallocal_IXT_cas)
        store_file(information_save_path+'/final_information_y.pkl',finallocal_ITY_cas)
#       =================================================================
        store_file(information_save_path+'/KDEinformation_x.pkl',KDElocal_IXT_cas)
        store_file(information_save_path+'/KDEinformation_y.pkl',KDElocal_ITY_cas)
        store_file(information_save_path+'/KDEfinal_information_x.pkl',KDEfinallocal_IXT_cas)
        store_file(information_save_path+'/KDEfinal_information_y.pkl',KDEfinallocal_ITY_cas)
#       ======================================================================
        store_file(information_save_path+'/Kdisinformation_x.pkl',Kdislocal_IXT_cas)
        store_file(information_save_path+'/Kdisinformation_y.pkl',Kdislocal_ITY_cas)
        store_file(information_save_path+'/Kdisfinal_information_x.pkl',Kdisfinallocal_IXT_cas)
        store_file(information_save_path+'/Kdisfinal_information_y.pkl',Kdisfinallocal_ITY_cas)
#       ==============================Save the performance===========================================
        accuarcy_save_path='history_accuarcy/acc_'+num_of_experment # The path of saving performance
        if not os.path.exists(accuarcy_save_path):
            os.makedirs(accuarcy_save_path)
        store_file(accuarcy_save_path+'/'+'acc.pkl',history)
        store_file(accuarcy_save_path+'/'+'output_history.pkl',output_history)
        print('history is saved')