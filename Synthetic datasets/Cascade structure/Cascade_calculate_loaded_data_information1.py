"The functions needed for calculating mutual information by three methods"
#=====================================================================================================
#                           The libraries we need
#=====================================================================================================
import numpy as np
import scipy.io as sio
import os
import pickle as cPickle
from keras.models import load_model
import tqdm
import time
import kde
import keras.backend as K
#==============================================================================================================================================
"The functions needed for calculate information"
def extract_probs(label, x):
    "Calculate the probabilities of the given data and labels p(x), p(y)"
    b = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    unique_array, unique_indices, unique_inverse_x, unique_counts = \
        np.unique(b, return_index=True, return_inverse=True, return_counts=True)
    pxs = unique_counts / float(np.sum(unique_counts))   # Calculate probabilities of X
    b_y = np.ascontiguousarray(label).view(np.dtype((np.void, label.dtype.itemsize * label.shape[1])))
    unique_array_y, unique_indices_y, unique_inverse_y, unique_counts_y = \
        np.unique(b_y, return_index=True, return_inverse=True, return_counts=True)
    pys1 = unique_counts_y / float(np.sum(unique_counts_y)) # Calculate probabilities of Y 
    return pys1,unique_inverse_x, unique_inverse_y, pxs
def calc_information_from_mat(px, py, ps2, data, unique_inverse_x, unique_inverse_y):
    "Calculate the mutual information based on binning of the data"
    H_T = -np.sum(ps2 * np.log2(ps2)) # H(T)
    H_TX = calc_condtion_entropy(px, data, unique_inverse_x) # H(T|X)
    H_TY = calc_condtion_entropy(py, data, unique_inverse_y) # H(T|Y)
    IY = H_T- H_TY   # I(Y;T)=H(T)-H(T|Y)
    IX = H_T - H_TX  # I(X;T)=H(T)-H(T|X)
    return IX, IY
def calc_entropy_for_specipic_t(current_ts, px_i):
    "Calculate entropy H(T|x_i) "
    b2 = np.ascontiguousarray(current_ts).view(
        np.dtype((np.void, current_ts.dtype.itemsize * current_ts.shape[1])))
    unique_array, unique_inverse_t, unique_counts = \
        np.unique(b2, return_index=False, return_inverse=True, return_counts=True)
    p_current_ts = unique_counts / float(sum(unique_counts))
    p_current_ts = np.asarray(p_current_ts, dtype=np.float64).T 
    H2X = px_i * (-np.sum(p_current_ts * np.log2(p_current_ts)))
    return H2X
def calc_condtion_entropy(px, t_data, unique_inverse_x):
    "Condition entropy of T given X"
    H2X_array = np.array([calc_entropy_for_specipic_t(t_data[unique_inverse_x == i, :], px[i]) for i in range(px.shape[0])])
    H_TX = np.sum(H2X_array)
    return  H_TX
def calculate_layer_mutual_information(layeroutput,PXs, PYs,unique_inverse_x,unique_inverse_y,bins):
    digitized=bins[np.digitize(np.squeeze(layeroutput.reshape(1, -1)), bins) - 1].reshape(len(layeroutput), -1) # Discrete T by binning into multi-intervals 
    b= np.ascontiguousarray(digitized).view(np.dtype((np.void, digitized.dtype.itemsize * digitized.shape[1]))) 
    unique_array, unique_inverse_t, unique_counts=np.unique(b, return_index=False, return_inverse=True, return_counts=True) # Unique  
    p_ts=unique_counts / float(sum(unique_counts))   # Calculate the distribution of T 
    local_IXT,local_ITY= calc_information_from_mat(PXs, PYs, p_ts, digitized, unique_inverse_x, unique_inverse_y) # Calculate the mutual information
    print('BINI(X;T) is:'+str(local_IXT),'BINI(Y;T) is:'+str(local_ITY)) 
    return local_IXT,local_ITY
def calculate_mutual_information(layeroutput,PXs, PYs,unique_inverse_x,unique_inverse_y):
    "Calculate mutual information based on uppper bound"
    noise_variance = 1e-3  # The variance of noise added in estimator according to Kolchinsky and Tracey, Nonlinear Information Bottleneck, 2017. Eq. 10
    h_upper=kde.entropy_estimator_kl(layeroutput, noise_variance) #Calculate marginal entropy based on KL divergence 
    h_lower=kde.entropy_estimator_bd(layeroutput, noise_variance) #Calculated marginal entropy based on Bhattacharyaa distance
    h_condition_on_data=kde.kde_condentropy(layeroutput, noise_variance) # Calculate conditional entropy of T given X based on assumption of Gaussian Distribution
    nats2bits = 1.0/np.log(2)  
    h_condition_on_label=0
    for i in range(len(PYs)): # Calculate conditional entropy of T given Y
        hcond_upper = kde.entropy_estimator_kl(layeroutput[unique_inverse_y==i],noise_variance)
        h_condition_on_label += PYs[i] * hcond_upper
    local_IXT=nats2bits*(h_upper-h_condition_on_data) # I(X;T)=H(T)-H(T|X)
    local_ITY=nats2bits*(h_upper-h_condition_on_label) # I(Y;T)=H(T)-H(T|Y)
    return local_IXT,local_ITY
def calculate_lower_mutual_information(layeroutput,PXs, PYs,unique_inverse_x,unique_inverse_y):
    "Calculate mutual information based on lower bound"
    noise_variance = 1e-3 
    h_lower=kde.entropy_estimator_bd(layeroutput, noise_variance) #Calculated marginal entropy based on Bhattacharyaa distance
    h_condition_on_data=kde.kde_condentropy(layeroutput, noise_variance)  # Calculate conditional entropy of T given X based on assumption of Gaussian Distribution
    nats2bits = 1.0/np.log(2) 
    h_condition_on_label=0
    for i in range(len(PYs)):   # Calculate conditional entropy of T given Y
        hcond_lower = kde.entropy_estimator_bd(layeroutput[unique_inverse_y==i],noise_variance)
        h_condition_on_label += PYs[i] * hcond_lower
    local_IXT=nats2bits*(h_lower-h_condition_on_data)  # I(X;T)=H(T)-H(T|X)
    local_ITY=nats2bits*(h_lower-h_condition_on_label)  # I(Y;T)=H(T)-H(T|Y)
    return local_IXT,local_ITY
        