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
import keras.backend as K
from keras.optimizers import Adam
from keras.models import Sequential
import os
import pickle as cPickle
import matplotlib.pyplot as plt
from keras.models import Model
from Cascade_calculate_loaded_data_information1 import calculate_layer_mutual_information,extract_probs,calculate_mutual_information,calculate_lower_mutual_information
# from mutual_information import mutual_information
import h5py
from keras.datasets import mnist
#================================================================================================
def load_generated_data_var(data_name):
    "load generated data"
    d=sio.loadmat(data_name)
    # d=h5py.File(data_name)
    print(d['F'].shape,d['y'].shape)
    data=dict()
    data['F']=[]
    data['y']=[]
    data['F']=np.array(d['F'])
    data['y']=np.array(d['y']).T
    return data
def load_generated_data(data_name):
    "load generated data"
#    d=sio.loadmat(data_name)
    d=h5py.File(data_name)
    print(d)
    data=dict()
    data['F']=[]
    data['y']=[]
    data['F']=np.array(d['F']).T
    data['y']=np.array(d['y'])
    return data
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
    stop_train_index = perc(percent_of_train[0], data_sets_org['F'].shape[0])
    start_test_index = stop_train_index
    if percent_of_train > min_test_data:
        start_test_index = perc(min_test_data, data_sets_org['F'].shape[0])
    data_sets.train = C()
    data_sets.test = C()
    if shuffle_data:
        shuffled_data, shuffled_labels = shuffle_in_unison_inplace(data_sets_org['F'], data_sets_org['y'])
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
        #=======================================get parameters=================================================
        self.hidden_parameters=dict()
        self.hidden_parameters['weights_norm']=[]   # L2 norm of weights
        self.hidden_parameters['gradmean']=[]  # Mean of gradients
        self.hidden_parameters['gradstd']=[]   # Std of gradients
    def on_epoch_end(self, epoch, logs={}):
        print('epoch number is:',epoch)
        self.local_IXT_epoch,self.local_ITY_epoch=[],[]
        self.KDElocal_IXT_epoch,self.KDElocal_ITY_epoch=[],[]
        self.Kdislocal_IXT_epoch,self.Kdislocal_ITY_epoch=[],[]
                #=======================================get parameters=================================================
        self.weights_norm=[]   # L2 norm of weights
        self.gradmean=[]   # Mean of gradients
        self.gradstd=[]   #
        input_tensors = [model.inputs[0], # input data
                 model.sample_weights[0], # how much to weight each sample by
                 model.targets[0], # labels
                 K.learning_phase(), # train or test mode
                 ]
        weights = [tensor for tensor in model.trainable_weights]
        gradients = model.optimizer.get_gradients(model.total_loss, weights) # gradient tensors
        get_gradients = K.function(inputs=input_tensors, outputs=gradients)
        inputs = [self.DATA, # X
                  np.ones(self.DATA.shape[0]), # sample weights
                  self.Label, # y
                  0 # learning phase in TEST mode
        ]
#        =========================================================
        gradient_value=get_gradients(inputs)
        weights_value=model.get_weights()
#        print('The gradient of the first layer is',gradient_value[0])
#        print('The weight of is:',weights_value)
        
#        ======================================================================
        for layernum in range(len(structure)):
        	#           ============================calculate parameters==============================================
            self.weights_norm.append(np.linalg.norm(weights_value[2*layernum]))
            # L2 norm of weights
            self.gradmean.append(np.mean(np.linalg.norm(gradient_value[2*layernum],axis=1))) # Mean of gradients
            self.gradstd.append(np.std(np.linalg.norm(gradient_value[2*layernum],axis=1))) #
            print('The gradient of layer',str(layernum),str(self.weights_norm))
            print('The weight norm mean of layer',str(layernum),str(self.gradmean))
            print('The weight norm std of layer',str(layernum),str(self.gradstd))
#            =====================================================================================================
            intermediate_layer_model = Model(inputs=self.modelToTrain.input,outputs=self.modelToTrain.get_layer(self.layer_name+str(layernum)).output)
            #model.summary()
            layeroutput=intermediate_layer_model.predict(self.DATA)
            # local_IXT=mutual_information((self.DATA,layeroutput[0:len(self.DATA)]), k=1)
            # local_ITY=mutual_information((layeroutput[0:len(self.DATA)],np.float64(Label)), k=4)
            # =========================for relu===================================
            # max_value=layeroutput.max()
            # bins = np.linspace(0, max_value,30)
            # bins = bins.astype(np.float32)
            # self.bins=bins
            # print('max value is',max_value)
            #=============================================================
            
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
        # self.out_log['each_final_layer'].append(finallayeroutput_cas)
#        self.out_log.append(layeroutput_cas)
        self.local_IXT_layer.append(self.local_IXT_epoch)
        self.local_ITY_layer.append(self.local_ITY_epoch)
        # ===================================================================================
        self.KDElocal_IXT_layer.append(self.KDElocal_IXT_epoch)
        self.KDElocal_ITY_layer.append(self.KDElocal_ITY_epoch)
        # ======================================================================================
        self.Kdislocal_IXT_layer.append(self.Kdislocal_IXT_epoch)
        self.Kdislocal_ITY_layer.append(self.Kdislocal_ITY_epoch)
        # =================================================================================
         # ===========================save parameters======================================================
        self.hidden_parameters['weights_norm'].append(self.weights_norm)
        self.hidden_parameters['gradmean'].append(self.gradmean)
        self.hidden_parameters['gradstd'].append(self.gradstd)
        print('The weight norm mean of epoch',str(epoch),str(self.hidden_parameters['gradmean']))
#==============================================================================================================================================
"The functions needed for creating network"
def connectOutputBlock(modelToConnectOut,outputsize,activation):
    "The final prediction block"
    modelToConnectOut.add(Dense(outputsize,activation=activation,name='Prediction_Layer'))
    return modelToConnectOut
def initialblock(structure_nodes,model,X_train,activation,layernum):
    "The initialization setting of the network"
    model.add(Dense(structure_nodes,activation=activation,input_dim=X_train.shape[1],kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.3, seed=None),name='Layer'+str(layernum)))#kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01),
    return model
def trainModel(modelToTrain,data,currentEpochs,batch_size,optimizer,accuarcy_name,filepath,structure,DATA,Label):
  modelToTrain.compile(loss='mean_squared_error',optimizer=optimizer,metrics=[accuarcy_name])
  bins = np.linspace(-1, 1,30)
  bins = bins.astype(np.float32)
  prediction_logs=OutputObserver(np.concatenate((data[0],data[2])),structure,DATA,Label,bins,modelToTrain)
  trainingResults = modelToTrain.fit(data[0],data[1], batch_size=batch_size, nb_epoch=currentEpochs, validation_data=(data[2],data[3]),verbose=1,callbacks=[prediction_logs])
  hist = trainingResults.history#keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=nb_epoch/10, verbose=1, mode='auto'),keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto'),keras.callbacks.ModelCheckpoint(filepath+"/weights-improvement-{epoch:02d}.hdf5", monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1),
  return hist,prediction_logs.out_log,prediction_logs.local_IXT_layer,prediction_logs.local_ITY_layer,prediction_logs
#========================The defination of the parameters we need==============================================================================
def parameters(number_of_data,num_of_train):
      "the structure can alter, e.g:list(map(int,np.linspace(2,100,20))) or [100,50,30,20,15,10]"
      input_dimension=12
      percent_of_train=np.array([80.00])
      structure=[5,3,1]# The structure  of the network (The number of nodes in each layer)
      outputsize=1          # The dimension of the prediction layer
      num_of_experment='check weights'+str(structure)+'_INFORMATION_'+str(percent_of_train)+'_'+'_full_size_experment_'+str(number_of_data)+'_'+str(num_of_train)  #for change the saving name of results
      name_num_of_experment=str(structure)+'_experment_'+str(number_of_data)
      outputsize=1 # The size of the output
      lr = 0.001  #learning rate
      optimizer=Adam(lr=lr) # The name of the optimizer
      batch_size=256 # The siz of the batch_size
      nb_epoch=3000 # The number of training epoch
      activation='tanh'#The activation function,tanh
      namecase=num_of_experment+'layer_output_case'
      accuarcy_name='acc'
      use_generated_structure=False
      return lr,optimizer,batch_size,nb_epoch,activation,namecase,structure,use_generated_structure,num_of_experment,outputsize,input_dimension,percent_of_train,accuarcy_name,name_num_of_experment
#=====================================================================================================================
#                load data and shuffled
#====================================================================
for number_of_data in ['dexter']:#['arcene','dexter','dorothea','Epileptic','gisette','LSVT','madelon']:#17
    for num_of_train in range(1):
        lr,optimizer,batch_size,nb_epoch,activation,namecase,structure,use_generated_structure,num_of_experment,outputsize,input_dimension,percent_of_train,accuarcy_name,name_num_of_experment=parameters(number_of_data,num_of_train)
        if number_of_data=='MNIST':
          (X_train1, Y_train1),(X_test1, Y_test1)=mnist.load_data()
          X_train1 = np.array(X_train1.reshape(-1,X_train1.shape[1]*X_train1.shape[2]))
          X_test1= np.array(X_test1.reshape(-1,X_test1.shape[1]*X_test1.shape[2]))
          Y_train1=Y_train1.reshape(-1,1)
          Y_test1=Y_test1.reshape(-1,1)
          index_train1=np.where(Y_train1==5)
          index_train2=np.where(Y_train1==6)
          index_test1=np.where(Y_test1==5) 
          index_test2=np.where(Y_test1==6)
          Y_train1[index_train1]=0
          Y_train1[index_train2]=1
          Y_test1[index_test1]=0
          Y_test1[index_test2]=1
          print('the task is binary classification:',X_train1[index_train1[0]].shape)
          X_train=np.concatenate((X_train1[index_train1[0]],X_train1[index_train2[0]]),axis=0)
          Y_train=np.concatenate((Y_train1[index_train1],Y_train1[index_train2]))
          X_test=np.concatenate((X_test1[index_test1[0]],X_test1[index_test2[0]]),axis=0)
          Y_test=np.concatenate((Y_test1[index_test1],Y_test1[index_test2]))
          print('the task is binary classification:',X_train.shape)
          print(Y_test)
          Data=np.concatenate((X_train,X_test))
          DATA=Data
          Label=np.concatenate((Y_train,Y_test))
          Label=Label.reshape(-1,1)
        elif number_of_data=='var_u':
          Data_set=load_generated_data_var('dataset/'+number_of_data+'.mat')
          Data,Label=Data_set['F'],Data_set['y']
          DATA=Data
          Data_set_shuffled=data_label_shuffle(Data_set, percent_of_train,shuffle_data=True)
          X_train, Y_train,X_test, Y_test=Data_set_shuffled.train.data,Data_set_shuffled.train.labels,Data_set_shuffled.test.data,Data_set_shuffled.test.labels
        else:
          Data_set=load_generated_data('dataset/'+number_of_data+'.mat')
          Data,Label=Data_set['F'],Data_set['y']
          DATA=Data
          Data_set_shuffled=data_label_shuffle(Data_set, percent_of_train,shuffle_data=True)
          X_train, Y_train,X_test, Y_test=Data_set_shuffled.train.data,Data_set_shuffled.train.labels,Data_set_shuffled.test.data,Data_set_shuffled.test.labels
        Dataset=[X_train, Y_train,X_test, Y_test]
        #===========================================================================================
        #                                        Cascade Training process
        #===========================================================================================
        print('Train end to end connected learning')
        history = dict()
        output_history=dict()
        new_data_set=[X_train, Y_train,X_test, Y_test]
        # ==================================================================================
        #==============================construct fully  connect network=======================================================
        localIXT,localITY=[],[]
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
        hist,prediction_logs_out,local_IXT_layer,local_ITY_layer,prediction_logs=trainModel(model,Dataset,nb_epoch,batch_size,optimizer,accuarcy_name,filepath,structure,DATA,Label)
        history['accuarcy'].append(hist)
        output_history['output'].append(prediction_logs_out)
        model_save_path='fully_model_saving/model_'+num_of_experment # The path of saving model
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        model.save(model_save_path+'/model.h5')
        print('fully_model is generated')
        # ==============================================================================
        #        ==============================save parameters===================================
        parameters_save_path='para information_save/'+num_of_experment
        if not os.path.exists(parameters_save_path):
            os.makedirs(parameters_save_path)
        store_file(parameters_save_path+'/weight_norm.pkl',prediction_logs.hidden_parameters['weights_norm'])
        store_file(parameters_save_path+'/gradient_mean.pkl',prediction_logs.hidden_parameters['gradmean'])
        store_file(parameters_save_path+'/gradient_std.pkl',prediction_logs.hidden_parameters['gradstd'])
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

