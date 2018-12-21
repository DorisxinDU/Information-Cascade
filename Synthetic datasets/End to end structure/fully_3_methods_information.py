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
from keras.models import Sequential
import os
import pickle as cPickle
from keras.models import Model
from calculate_loaded_data_information import calculate_layer_mutual_information,extract_probs,calculate_mutual_information,calculate_lower_mutual_information
from training_parameters import parameters # Import parametrs used for training model
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
class OutputObserver(keras.callbacks.Callback):
    'callback to observe the output of the network'
    def __init__(self, data_set,structure,DATA,Label,bins,modelToTrain,logs={}):
        self.out_log = dict()
#        self.out_log['eachlayer']=[]
        self.out_log['final_layer']=[]
        self.modelToTrain=modelToTrain
        self.data_set = data_set
        self.layer_name='Layer'
        self.structure=structure
        self.DATA=DATA
        self.Label=Label
        self.bins=bins
        self.pys1,self.unique_inverse_x,self. unique_inverse_y,self. pxs = extract_probs(self.Label,self.DATA)
        self.PXs, self.PYs = np.asarray(self.pxs).T, np.asarray(self.pys1).T
        self.local_IXT_layer,self.local_ITY_layer=[],[]
        self.KDElocal_IXT_layer,self.KDElocal_ITY_layer=[],[]
        self.Kdislocal_IXT_layer,self.Kdislocal_ITY_layer=[],[]
        self.out_log['layer'+str(len(structure))]=[]
        for layernum in range(len(structure)):
            self.out_log['layer'+str(layernum)]=[]
    def on_epoch_end(self, epoch, logs={}):
        print('epoch number is:',epoch)
        self.local_IXT_epoch,self.local_ITY_epoch=[],[]
        self.KDElocal_IXT_epoch,self.KDElocal_ITY_epoch=[],[]
        self.Kdislocal_IXT_epoch,self.Kdislocal_ITY_epoch=[],[]
        for layernum in range(len(structure)):
            intermediate_layer_model = Model(inputs=self.modelToTrain.input,outputs=self.modelToTrain.get_layer(self.layer_name+str(layernum)).output)
            #model.summary()
            layeroutput=intermediate_layer_model.predict(self.DATA)
            local_IXT,local_ITY=calculate_layer_mutual_information(layeroutput[0:len(self.DATA)],self.PXs,self.PYs,self.unique_inverse_x,self.unique_inverse_y,self.bins)
            self.local_IXT_epoch.append(local_IXT)
            self.local_ITY_epoch.append(local_ITY)
            # ========================================================================================================
            KDElocal_IXT,KDElocal_ITY=calculate_mutual_information(layeroutput[0:len(self.DATA)],self.PXs,self.PYs,self.unique_inverse_x,self.unique_inverse_y,self.bins)
            print('KDEI(X;T) is:'+str(KDElocal_IXT),'KDEI(Y;T) is:'+str(KDElocal_ITY))
            self.KDElocal_IXT_epoch.append(KDElocal_IXT)
            self.KDElocal_ITY_epoch.append(KDElocal_ITY)
            # =============================================================================================================
            Kdislocal_IXT,Kdislocal_ITY=calculate_lower_mutual_information(layeroutput[0:len(self.DATA)],self.PXs,self.PYs,self.unique_inverse_x,self.unique_inverse_y,self.bins)
            print('KdisI(X;T) is:'+str(Kdislocal_IXT),'KdisI(Y;T) is:'+str(Kdislocal_ITY))
            self.Kdislocal_IXT_epoch.append(Kdislocal_IXT)
            self.Kdislocal_ITY_epoch.append(Kdislocal_ITY)
            # ===================================================================================================
            if layernum==len(structure)-1:
                intermediate_predict_layer_model = Model(inputs=self.modelToTrain.input,outputs=self.modelToTrain.get_layer('Prediction_Layer').output)
                finallayeroutput=intermediate_predict_layer_model.predict(self.DATA)
                self.out_log['final_layer'].append(finallayeroutput)
                local_IXT,local_ITY=calculate_layer_mutual_information(finallayeroutput[0:len(self.DATA)],self.PXs,self.PYs,self.unique_inverse_x,self.unique_inverse_y,self.bins)
                self.local_IXT_epoch.append(local_IXT)
                self.local_ITY_epoch.append(local_ITY)
                # ==========================================================================================================
                KDElocal_IXT,KDElocal_ITY=calculate_mutual_information(finallayeroutput[0:len(self.DATA)],self.PXs,self.PYs,self.unique_inverse_x,self.unique_inverse_y,self.bins)
                print('KDEI(X;T) is:'+str(KDElocal_IXT),'KDEI(Y;T) is:'+str(KDElocal_ITY))
                self.KDElocal_IXT_epoch.append(KDElocal_IXT)
                self.KDElocal_ITY_epoch.append(KDElocal_ITY)
                # ==========================================================================================================
                Kdislocal_IXT,Kdislocal_ITY=calculate_lower_mutual_information(finallayeroutput[0:len(self.DATA)],self.PXs,self.PYs,self.unique_inverse_x,self.unique_inverse_y,self.bins)
                print('KdisI(X;T) is:'+str(Kdislocal_IXT),'KdisI(Y;T) is:'+str(Kdislocal_ITY))
                self.Kdislocal_IXT_epoch.append(Kdislocal_IXT)
                self.Kdislocal_ITY_epoch.append(Kdislocal_ITY)
                self.out_log['layer'+str(len(structure))].append(finallayeroutput)
                del intermediate_predict_layer_model, finallayeroutput 
                # ==========================================================================================================
            self.out_log['layer'+str(layernum)].append(layeroutput)
            del intermediate_layer_model,layeroutput 
        self.local_IXT_layer.append(self.local_IXT_epoch)
        self.local_ITY_layer.append(self.local_ITY_epoch)
        # ===================================================================================
        self.KDElocal_IXT_layer.append(self.KDElocal_IXT_epoch)
        self.KDElocal_ITY_layer.append(self.KDElocal_ITY_epoch)
        # ======================================================================================
        self.Kdislocal_IXT_layer.append(self.Kdislocal_IXT_epoch)
        self.Kdislocal_ITY_layer.append(self.Kdislocal_ITY_epoch)
        # ================================================================================
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
def trainModel(modelToTrain,data,currentEpochs,batch_size,optimizer,accuarcy_name,filepath,structure,DATA,Label):
  modelToTrain.compile(loss='mean_squared_error',optimizer=optimizer,metrics=[accuarcy_name])
  bins = np.linspace(-1, 1,25)
  bins = bins.astype(np.float32)
  prediction_logs=OutputObserver(np.concatenate((data[0],data[2])),structure,DATA,Label,bins,modelToTrain)
  trainingResults = modelToTrain.fit(data[0],data[1], batch_size=batch_size, nb_epoch=currentEpochs, validation_data=(data[2],data[3]),verbose=1,callbacks=[prediction_logs])
  hist = trainingResults.history
  return hist,prediction_logs.out_log,prediction_logs.local_IXT_layer,prediction_logs.local_ITY_layer,prediction_logs
for number_of_data in [17]: 
    for num_of_train in range(1):
        lr,optimizer,batch_size,nb_epoch,activation,structure,num_of_experment,outputsize,percent_of_train,accuarcy_name,name_num_of_experment=parameters(number_of_data,num_of_train)
        Data_set=load_generated_data('data_saving/'+name_num_of_experment+'/dataset.mat')
        Data,Label=Data_set['data_sets.data'],Data_set['data_sets.label']
        DATA=Data
        Data_set_shuffled=data_label_shuffle(Data_set, percent_of_train,shuffle_data=True)
        X_train, Y_train,X_test, Y_test=Data_set_shuffled.train.data,Data_set_shuffled.train.labels,Data_set_shuffled.test.data,Data_set_shuffled.test.labels
        Dataset=[X_train, Y_train,X_test, Y_test]
        #===========================================================================================
        #                                        Training process
        #===========================================================================================
        print('Train end to end connected learning')
        history = dict()
        output_history=dict()
        new_data_set=[X_train, Y_train,X_test, Y_test]
        # ==================================================================================
        #==============================construct end to end network=======================================================
        localIXT,localITY=[],[]
        for layernum in range(len(structure)):
              if layernum==0:
                  model = Sequential()
                  model=initialblock(structure[layernum],model,X_train,activation,layernum)
              else:
                  model.add(Dense(structure[layernum],activation=activation,name='Layer'+str(layernum)))
        model=connectOutputBlock(model,outputsize,activation)
        model.summary()
        # ++++++++++++++++++++++++++++++++++++++++++++Training model++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        filepath='check_point_model_layer_save/model_'+num_of_experment
        # The path of saving model_each_epoch
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        history['accuarcy'] = []
        output_history['output'] = []
        hist,prediction_logs_out,local_IXT_layer,local_ITY_layer,prediction_logs=trainModel(model,Dataset,nb_epoch,batch_size,optimizer,accuarcy_name,filepath,structure,DATA,Label)
        history['accuarcy'].append(hist)
        output_history['output'].append(prediction_logs_out)
        model_save_path='fully_model_saving/model_'+num_of_experment # The path of saving model
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        model.save(model_save_path+'/model.h5')
        print('fully_model is generated')
        # ==============================================================================
        localIXT=local_IXT_layer
        localITY=local_ITY_layer
        information_save_path='information_save/'+num_of_experment
        if not os.path.exists(information_save_path):
            os.makedirs(information_save_path)
        store_file(information_save_path+'/information_x.pkl',localIXT)
        store_file(information_save_path+'/information_y.pkl',localITY)
        # =================================================================================
        KDElocalIXT=prediction_logs.KDElocal_IXT_layer
        KDElocalITY=prediction_logs.KDElocal_ITY_layer
        information_save_path='information_save/'+num_of_experment
        if not os.path.exists(information_save_path):
            os.makedirs(information_save_path)
        store_file(information_save_path+'/KDEinformation_x.pkl',KDElocalIXT)
        store_file(information_save_path+'/KDEinformation_y.pkl',KDElocalITY)
        # ====================================================================================
        KdislocalIXT=prediction_logs.Kdislocal_IXT_layer
        KdislocalITY=prediction_logs.Kdislocal_ITY_layer
        information_save_path='information_save/'+num_of_experment
        if not os.path.exists(information_save_path):
            os.makedirs(information_save_path)
        store_file(information_save_path+'/Kdisinformation_x.pkl',KdislocalIXT)
        store_file(information_save_path+'/Kdisinformation_y.pkl',KdislocalITY)

        #==============================plot training process=======================================================
        accuarcy_save_path='history_accuarcy/acc_'+num_of_experment
        if not os.path.exists(accuarcy_save_path):
            os.makedirs(accuarcy_save_path)
        store_file(accuarcy_save_path+'/'+'acc.pkl',history)
        store_file(accuarcy_save_path+'/'+'output_history.pkl',output_history)
        print('history is saved')
