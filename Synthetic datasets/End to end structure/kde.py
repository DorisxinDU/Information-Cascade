import keras
import keras.backend as K
import tensorflow as tf
import numpy as np
import scipy
def Kget_dists(X):
    """Keras code to compute the pairwise distance matrix for a set of
    vectors specifie by the matrix X.
    """
    # X=tf.convert_to_tensor(X)
    x2 = np.expand_dims(np.sum(np.square(X), axis=1), 1)
    dists = x2 + np.transpose(x2) - 2*np.dot(X, np.transpose(X))
    return dists#K.eval()

def get_shape(x):
    [dims,N]=np.shape(x)
    # dims = K.cast( K.shape(x)[1], K.floatx() ) 
    # N    = K.cast( K.shape(x)[0], K.floatx() )
    # print(dims,N)#K.eval(dims),K.eval(N)
    return dims, N

def entropy_estimator_kl(x, var):
    # KL-based upper bound on entropy of mixture of Gaussians with covariance matrix var * I 
    #  see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
    #  and Kolchinsky and Tracey, Nonlinear Information Bottleneck, 2017. Eq. 10
    # x=tf.convert_to_tensor(x)
    N,dims = get_shape(x)
    dists = Kget_dists(x)
    dists2 = dists / (2*var)
    normconst = (dims/2.0)*np.log(2*np.pi*var)
    lprobs =scipy.misc.logsumexp(-dists2, axis=1) - np.log(N) - normconst
    h = -np.mean(lprobs)
    return dims/2 + h#K.eval(dims/2 + h)

def entropy_estimator_bd(x, var):
    # Bhattacharyya-based lower bound on entropy of mixture of Gaussians with covariance matrix var * I 
    #  see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
    N,dims= get_shape(x)
    val = entropy_estimator_kl(x,4*var)
    return val + np.log(0.25)*dims/2#K.eval(val + np.log(0.25)*dims/2)

def kde_condentropy(output, var):
    # Return entropy of a multivariate Gaussian, in nats
    dims = output.shape[1]
    return (dims/2.0)*(np.log(2*np.pi*var) + 1)

