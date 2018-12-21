# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 09:36:59 2018

@author: TOSHIBA
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 11:46:26 2018

@author: xd3y15
"""

#=====================================================================================================
#                           The libraries we need
#=====================================================================================================
#=============reduce the storage==============================
"For avoiding programme using unecessary memory"
# import tensorflow as tf
# config=tf.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)
#==============================================================
import numpy as np
import scipy.io as sio
import os
import pickle as cPickle
from keras.models import load_model
#from keras.models import Model
import tqdm
import time
# from mutual_information import mutual_information
import kde
import keras.backend as K
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
"The functions needed for calculate information"
def extract_probs(label, x):
    """calculate the probabilities of the given data and labels p(x), p(y) and p(y|x)"""
    b = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    unique_array, unique_indices, unique_inverse_x, unique_counts = \
        np.unique(b, return_index=True, return_inverse=True, return_counts=True)
    pxs = unique_counts / float(np.sum(unique_counts))
    b_y = np.ascontiguousarray(label).view(np.dtype((np.void, label.dtype.itemsize * label.shape[1])))
    unique_array_y, unique_indices_y, unique_inverse_y, unique_counts_y = \
        np.unique(b_y, return_index=True, return_inverse=True, return_counts=True)
    pys1 = unique_counts_y / float(np.sum(unique_counts_y))
    return pys1,unique_inverse_x, unique_inverse_y, pxs
def calc_information_from_mat(px, py, ps2, data, unique_inverse_x, unique_inverse_y):
    """Calculate the MI based on binning of the data"""

    H2 = -np.sum(ps2 * np.log2(ps2))
#    print('the h2 is :',H2)
    H2X = calc_condtion_entropy(px, data, unique_inverse_x)
    H2Y = calc_condtion_entropy(py, data, unique_inverse_y)
    IY = H2 - H2Y
    IX = H2 - H2X
#    print('The information calculated is IX:'+str(IX),'IY:'+str(IY))
    return IX, IY
def calc_entropy_for_specipic_t(current_ts, px_i):
    """Calc entropy for specipic t"""
    b2 = np.ascontiguousarray(current_ts).view(
        np.dtype((np.void, current_ts.dtype.itemsize * current_ts.shape[1])))
    unique_array, unique_inverse_t, unique_counts = \
        np.unique(b2, return_index=False, return_inverse=True, return_counts=True)
    p_current_ts = unique_counts / float(sum(unique_counts))
    p_current_ts = np.asarray(p_current_ts, dtype=np.float64).T
    H2X = px_i * (-np.sum(p_current_ts * np.log2(p_current_ts)))
    return H2X
def calc_condtion_entropy(px, t_data, unique_inverse_x):
    # Condition entropy of t given x
    H2X_array = np.array([calc_entropy_for_specipic_t(t_data[unique_inverse_x == i, :], px[i]) for i in range(px.shape[0])])
    H2X = np.sum(H2X_array)
    return H2X
def calculate_layer_mutual_information(layeroutput,PXs, PYs,unique_inverse_x,unique_inverse_y,bins):
    digitized=bins[np.digitize(np.squeeze(layeroutput.reshape(1, -1)), bins) - 1].reshape(len(layeroutput), -1)
    b= np.ascontiguousarray(digitized).view(np.dtype((np.void, digitized.dtype.itemsize * digitized.shape[1])))
    unique_array, unique_inverse_t, unique_counts=np.unique(b, return_index=False, return_inverse=True, return_counts=True)
    p_ts=unique_counts / float(sum(unique_counts))
    local_IXT,local_ITY= calc_information_from_mat(PXs, PYs, p_ts, digitized, unique_inverse_x, unique_inverse_y)
    print('BINI(X;T) is:'+str(local_IXT),'BINI(Y;T) is:'+str(local_ITY))
    return local_IXT,local_ITY
def calculate_mutual_information(layeroutput,PXs, PYs,unique_inverse_x,unique_inverse_y,bins):
    noise_variance = 1e-3 
    # Compute marginal entropies
    h_upper=kde.entropy_estimator_kl(layeroutput, noise_variance)
    h_lower=kde.entropy_estimator_bd(layeroutput, noise_variance)
    h_condition_on_data=kde.kde_condentropy(layeroutput, noise_variance)
    nats2bits = 1.0/np.log(2) 
    h_condition_on_label=0
    for i in range(len(PYs)):
        hcond_upper = kde.entropy_estimator_kl(layeroutput[unique_inverse_y==i],noise_variance)
        h_condition_on_label += PYs[i] * hcond_upper
    local_IXT=nats2bits*(h_upper-h_condition_on_data)
    local_ITY=nats2bits*(h_upper-h_condition_on_label)
    # init = tf.global_variables_initializer()
    # sess = tf.Session()
    # sess.run(init)
    # local_IXT=local_IXT.eval(sess)
    # local_ITY=local_ITY.eval(sess)
    # with sess.as_default():
    #     local_IXT=nats2bits*(h_upper-h_condition_on_data).eval()
    #     local_ITY=nats2bits*(h_upper-h_condition_on_label).eval()
    # print('KDEI(X;T) is:',local_IXT,'KDEI(Y;T) is:',local_ITY)
    return local_IXT,local_ITY
def calculate_lower_mutual_information(layeroutput,PXs, PYs,unique_inverse_x,unique_inverse_y,bins):
    noise_variance = 1e-3 
    # Compute marginal entropies
#    h_upper=kde.entropy_estimator_kl(layeroutput, noise_variance)
    h_lower=kde.entropy_estimator_bd(layeroutput, noise_variance)
    h_condition_on_data=kde.kde_condentropy(layeroutput, noise_variance)
    nats2bits = 1.0/np.log(2) 
    h_condition_on_label=0
    for i in range(len(PYs)):
        hcond_lower = kde.entropy_estimator_bd(layeroutput[unique_inverse_y==i],noise_variance)
        h_condition_on_label += PYs[i] * hcond_lower
    local_IXT=nats2bits*(h_lower-h_condition_on_data)
    local_ITY=nats2bits*(h_lower-h_condition_on_label)
    # init = tf.global_variables_initializer()
    # sess = tf.Session()
    # sess.run(init)
    # local_IXT=local_IXT.eval(sess)
    # local_ITY=local_ITY.eval(sess)
#    print('KDEI(X;T) is:'+str(local_IXT),'KDEI(Y;T) is:'+str(local_ITY))
    # with sess.as_default():
        # local_IXT=nats2bits*(h_lower-h_condition_on_data).eval()
        # local_ITY=nats2bits*(h_lower-h_condition_on_label).eval()
    return local_IXT,local_ITY
#========================The defination of the parameters we need==============================================================================
def parameters(number_of_data,num_of_train):
      "the structure can alter, e.g:list(map(int,np.linspace(2,100,20))) or [100,50,30,20,15,10]"
#      input_dimension=12
      percent_of_train=np.array([100.00])
      structure=[10,7,5,3]# The structure  of the network (The number of nodes in each layer)
      outputsize=1          # The dimension of the prediction layer
      num_of_experment='test_save_cascade_learning'+str(percent_of_train)+'_'+str(structure)+'_experment_'+str(number_of_data)+'_'+str(num_of_train)  #for change the saving name of results
      name_num_of_experment=str(structure)+'_experment_'+str(number_of_data)
      outputsize=1 # The size of the output
      nb_epoch=1000 # The number of training epoch
      num_of_bins=30
      bins = np.linspace(-1, 1, num_of_bins)
      bins = bins.astype(np.float32)
      return nb_epoch,structure,num_of_experment,outputsize,percent_of_train,name_num_of_experment,bins
#=====================================================================================================================
#                load data and shuffled
#====================================================================


##======================================================================
##for number_of_data in [49]:
##    for num_of_train in range(1):
##        nb_epoch,structure,num_of_experment,outputsize,percent_of_train,name_num_of_experment,bins=parameters(number_of_data,num_of_train)
#        Data_set=load_generated_data('data_saving/'+name_num_of_experment+'/dataset.mat')
#        Data,Label=Data_set['data_sets.data'],Data_set['data_sets.label']
#        DATA=Data
#        Data_set_shuffled=data_label_shuffle(Data_set, percent_of_train,shuffle_data=True)
#        X_train, Y_train,X_test, Y_test=Data_set_shuffled.train.data,Data_set_shuffled.train.labels,Data_set_shuffled.test.data,Data_set_shuffled.test.labels
#        Dataset=[X_train, Y_train,X_test, Y_test]
#        pys1,unique_inverse_x, unique_inverse_y, pxs = extract_probs(Label,Data)
#        PXs, PYs = np.asarray(pxs).T, np.asarray(pys1).T
#        accuarcy_save_path='history_accuarcy/acc_'+num_of_experment
#        HISTORY=load_DATA(accuarcy_save_path+'/acc.pkl')
#        Outputs=load_DATA(accuarcy_save_path+'/output_history.pkl')
#        local_IXT_cas, local_ITY_cas,finallocal_IXT_cas, finallocal_ITY_cas=[],[],[],[]
#        for layernum in range(len(structure)):
#            finallocal_IXT_cas_layer, finallocal_ITY_cas_layer,local_IXT_cas_layer, local_ITY_cas_layer=[],[],[],[]
#            layer_name ='Layer'+str(layernum)
#            print('The current layer is: Layer_'+str(layernum))
#            time.sleep(0.01)
#            for epoch_num in tqdm.tqdm(range (nb_epoch*(layernum+1))):             
#                filepath='check_point_model_layer_save/model_'+num_of_experment+'/layer_'+str(layernum)+"/weights-improvement-{:02d}".format(epoch_num+1)+'.hdf5'
#                model=load_model(filepath)
##                model.summary()
#                layeroutput_cas=Outputs['iter'+str(layernum)][0]['eachlayer'][epoch_num][0:len(Data)]
#                finallayeroutput_cas=Outputs['iter'+str(layernum)][0]['each_final_layer'][epoch_num][0:len(Data)]
#                IX_CAS=mutual_information((DATA,layeroutput_cas), k=4)
#                IY_CAS=mutual_information((layeroutput_cas,np.float64(Label)), k=4)
#                finalIX_CAS=mutual_information((DATA,finallayeroutput_cas), k=4)
#                finalIY_CAS=mutual_information((finallayeroutput_cas,np.float64(Label)), k=4)
#                local_IXT_cas_layer.append(IX_CAS) 
#                local_ITY_cas_layer.append(IY_CAS)
#                finallocal_IXT_cas_layer.append(finalIX_CAS) 
#                finallocal_ITY_cas_layer.append(finalIY_CAS)
#                del finalIX_CAS,finalIY_CAS,IX_CAS,IY_CAS,model
#                if epoch_num==len(HISTORY['iter'+str(layernum)][0]['acc'])-1:
#                    Data=layeroutput_cas
#                    break 
#            local_IXT_cas.append(local_IXT_cas_layer)
#            local_ITY_cas.append(local_ITY_cas_layer)
#            finallocal_IXT_cas.append(finallocal_IXT_cas_layer)
#            finallocal_ITY_cas.append(finallocal_ITY_cas_layer)
#        information_save_path='cascade_information_save/'+num_of_experment
#        if not os.path.exists(information_save_path):
#            os.makedirs(information_save_path)
#        store_file(information_save_path+'/information_x.pkl',local_IXT_cas)
#        store_file(information_save_path+'/information_y.pkl',local_ITY_cas)
#        store_file(information_save_path+'/final_information_x.pkl',finallocal_IXT_cas)
#        store_file(information_save_path+'/final_information_y.pkl',finallocal_ITY_cas)
#            
#                
                
                
                
        