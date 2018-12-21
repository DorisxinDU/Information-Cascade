#========================The defination of the parameters we need==============================================================================
import numpy as np
from keras.optimizers import Adam
def parameters(number_of_data,num_of_train):
      "the structure can alter, e.g:list(map(int,np.linspace(2,100,20))) or [100,50,30,20,15,10]"
      percent_of_train=np.array([80.00])
      structure=[10,7,5,3]# The structure  of the network (The number of nodes in each layer)
      outputsize=1          # The dimension of the prediction layer
      num_of_experment='additivelayer_version_'+str(percent_of_train)+'_'+str(structure)+'_full_size_experment_'+str(number_of_data)+'_'+str(num_of_train)  #for change the saving name of results
      name_num_of_experment=str([10,7,5,3])+'_experment_'+str(number_of_data)
      outputsize=1 # The size of the output
      lr = 0.001  #learning rate
      optimizer=Adam(lr=lr) # The name of the optimizer
      batch_size=256 # The siz of the batch_size
      nb_epoch=2 # The number of training epoch
      activation='tanh'#The activation function
      accuarcy_name='acc'
      return lr,optimizer,batch_size,nb_epoch,activation,structure,num_of_experment,outputsize,percent_of_train,accuarcy_name,name_num_of_experment
#=====================================================================================================================