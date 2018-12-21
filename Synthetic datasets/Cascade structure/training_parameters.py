import numpy as np
from keras.optimizers import Adam
def parameters(number_of_data,num_of_train):
      "The parameters needed to define before training"
      percent_of_train=np.array([80.00]) 
      structure=[10,7,5,3]  # The structure  of the network (The number of nodes in each layer), if you use different datasets, the corresponding structure is different
      outputsize=1          # The dimension of the prediction layer
      num_of_experment='compression test'+str(percent_of_train)+'_'+str(structure)+'_experment_'+str(number_of_data)+'_'+str(num_of_train)  #For change the saving name of results with respect to different datasets and structures
      name_num_of_experment=str(structure)+'_experment_'+str(number_of_data) # The name of Input location for loading datasets
      lr = 0.001            #The initial learning rate
      optimizer=Adam(lr=lr) # The name of the optimizer
      batch_size=256        # The siz of the batch_size
      nb_epoch=1        # The number of training epoch
      activation='tanh'     #The activation function
      accuarcy_name='acc'   #The name of saving accuarcy
      return lr,optimizer,batch_size,nb_epoch,activation,structure,num_of_experment,outputsize,percent_of_train,accuarcy_name,name_num_of_experment