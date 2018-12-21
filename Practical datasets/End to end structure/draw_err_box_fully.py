import numpy as np
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import pickle as cPickle
import os
numDists = 5
randomDists = ['Fully connect\ntrain', 'Fully connect\ntest',' Cascade all\nlayers train', ' Cascade all\nlayers test','Single layer\ntrain', 'Single layer\ntest','Cascade second\nlayer train','Cascade second\nlayer test','Cascade third\n layer train','Cascade third\n layer test']
# ==============================================================================
def load_DATA(name):
    "Load history"
    with open(name, 'rb') as f:
        data=cPickle.load(f)
        return data
# ____________________________________
def parameters(number_of_data,num_of_train):
      "the structure can alter, e.g:list(map(int,np.linspace(2,100,20))) or [100,50,30,20,15,10]"
      input_dimension=12
      percent_of_train=np.array([70.00])
      structure=[15,10,7,3]# The structure  of the network (The number of nodes in each layer)
      outputsize=1          # The dimension of the prediction layer
      num_of_experment=str(structure)+'_regularizers'+str(percent_of_train)+'_full_size_experment_'+str(number_of_data)+'_'+str(num_of_train)   #for change the saving name of results
      name_num_of_experment=str(structure)+'_experment_'+str(number_of_data)
      outputsize=1 # The size of the output
      lr = 0.001  #learning rate
      
      batch_size=256 # The siz of the batch_size
      nb_epoch=30000 # The number of training epoch
      activation='tanh'#The activation function
      namecase=num_of_experment+'layer_output_case'
      accuarcy_name='acc'
      use_generated_structure=False
      return batch_size,nb_epoch,activation,namecase,structure,use_generated_structure,num_of_experment,outputsize,input_dimension,percent_of_train,accuarcy_name,name_num_of_experment
# ==============================================================================
for number_of_data in ['Epileptic']:#['arcene','dexter','dorothea','Epileptic','gisette','LSVT','madelon']:#['Epileptic']:#:
    for num_of_train in range(1):
        N=10
        batch_size,nb_epoch,activation,namecase,structure,use_generated_structure,num_of_experment,outputsize,input_dimension,percent_of_train,accuarcy_name,name_num_of_experment=parameters(number_of_data,num_of_train)
# ======================load accuarcy of fully connected========================================================
        fold_num=10
        history_each_fold=dict()
        history_each_fold['fully_connect_train']=[]
        history_each_fold['fully_connect_test']=[]
        history_each_fold['fully_cascade_train']=[]
        history_each_fold['fully_cascade_test']=[]
        history_each_fold['cascade_layer1_train']=[]
        history_each_fold['cascade_layer1_test']=[]
        history_each_fold['cascade_layer2_train']=[]
        history_each_fold['cascade_layer2_test']=[]
        history_each_fold['cascade_layer3_train']=[]
        history_each_fold['cascade_layer3_test']=[]
        num_of_experment1=num_of_experment
        num_of_experment2=str(structure)+'regular_stopping'+str(percent_of_train)+'_experment_'+str(number_of_data)+'_'+str(num_of_train) 
        finalepo=10
        for fold in range(fold_num):
            num_of_experment='fold_'+str(fold+1)+num_of_experment1
            accuarcy_save_path='history_accuarcy/acc_'+num_of_experment
            HISTORY=load_DATA(accuarcy_save_path+'/'+'acc.pkl')
            accuarcy_save_path='history_accuarcy/acc_'+num_of_experment
            print(np.shape(HISTORY['accuarcy'][-1][accuarcy_name]))
            history_each_fold['fully_connect_train'].append(np.mean(HISTORY['accuarcy'][-1][accuarcy_name][-finalepo:-1]))#np.mean(HISTORY['accuarcy'][-1][accuarcy_name][-30:-1])
            history_each_fold['fully_connect_test'].append(np.mean(HISTORY['accuarcy'][-1]['val_'+accuarcy_name][-finalepo:-1]))
            cascade_accuarcy_save_path='/home/xd3y15/real_world_dataset/cascade_ learning/history_accuarcy/acc_'+'fold_'+str(fold+1)+num_of_experment2
            cas_HISTORY=load_DATA(cascade_accuarcy_save_path+'/'+'acc.pkl')
            history_each_fold['fully_cascade_train'].append(np.mean(cas_HISTORY['iter'+str(2)][-1][accuarcy_name][-finalepo:-1]))
            history_each_fold['fully_cascade_test'].append(np.mean(cas_HISTORY['iter'+str(2)][-1]['val_'+accuarcy_name][-finalepo:-1]))
            history_each_fold['cascade_layer1_train'].append(np.mean(cas_HISTORY['iter'+str(0)][-1][accuarcy_name][-finalepo:-1]))
            history_each_fold['cascade_layer1_test'].append(np.mean(cas_HISTORY['iter'+str(0)][-1]['val_'+accuarcy_name][-finalepo:-1]))
            history_each_fold['cascade_layer2_train'].append(np.mean(cas_HISTORY['iter'+str(1)][-1][accuarcy_name][-finalepo:-1]))
            history_each_fold['cascade_layer2_test'].append(np.mean(cas_HISTORY['iter'+str(1)][-1]['val_'+accuarcy_name][-finalepo:-1]))
            history_each_fold['cascade_layer3_train'].append(np.mean(cas_HISTORY['iter'+str(2)][-1][accuarcy_name][-finalepo:-1]))
            history_each_fold['cascade_layer3_test'].append(np.mean(cas_HISTORY['iter'+str(2)][-1]['val_'+accuarcy_name][-finalepo:-1]))
            # history_each_fold['cascade_layer3_train']=cas_HISTORY['iter'+str(2)][-1][accuarcy_name]
            # history_each_fold['cascade_layer3_test']=cas_HISTORY['iter'+str(2)][-1]['val_'+accuarcy_name]
# ======================load accuarcy of cascade each layer connected========================================================
# Generate some random indices that we'll use to resample the original data
# arrays. For code brevity, just use the same random indices for each array

    fully_connect_train= history_each_fold['fully_connect_train']
    print(fully_connect_train,'the ten fold')
    fully_connect_test=history_each_fold['fully_connect_test']
    cascade_full_train=history_each_fold['fully_cascade_train']
    cascade_full_test=history_each_fold['fully_cascade_test'] 
    cascade_first_train= history_each_fold['cascade_layer1_train']
    cascade_first_test= history_each_fold['cascade_layer1_test']
    cascade_second_train= history_each_fold['cascade_layer2_train']
    cascade_second_test= history_each_fold['cascade_layer2_test']
    cascade_third_train= history_each_fold['cascade_layer3_train']
    cascade_third_test= history_each_fold['cascade_layer3_test']

    data = [fully_connect_train,fully_connect_test,cascade_full_train,cascade_full_test,cascade_first_train,cascade_first_test,cascade_second_train,cascade_second_test,cascade_third_train,cascade_third_test]#,cascade_third_train,cascade_third_test]
    print('this is the data:',np.shape(data))
    fig, ax1 = plt.subplots(figsize=(20, 12))
    fig.canvas.set_window_title('A Boxplot Example')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.33)

    bp = ax1.boxplot(data,0, '')#, notch=0,vert=1, whis=1.5
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)

    # Hide these grid behind plot objects
    ax1.set_axisbelow(True)
    ax1.set_title('Comparison of different models performance on train and test sets',fontsize=15)
    ax1.set_xlabel('Training Model',fontsize=28)
    ax1.set_ylabel('Accuarcy Value',fontsize=28)

    # Now fill the boxes with desired colors
    boxColors = ['darkkhaki', 'royalblue']
    numBoxes = numDists*2
    medians = list(range(numBoxes))
    for i in range(numBoxes):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        boxCoords = np.column_stack([boxX, boxY])
        # Alternate between Dark Khaki and Royal Blue
        k = i % 2
        boxPolygon = Polygon(boxCoords, facecolor=boxColors[k])
        ax1.add_patch(boxPolygon)
        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        medianX = []
        medianY = []
        for j in range(2):
            medianX.append(med.get_xdata()[j])
            medianY.append(med.get_ydata()[j])
            ax1.plot(medianX, medianY, 'k')
            medians[i] = medianY[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        ax1.plot([np.average(med.get_xdata())], [np.average(data[i])],
                 color='b', marker='*', markeredgecolor='k')

    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, numBoxes + 0.5)
    top = 1.1
    bottom = 0.7
    ax1.set_ylim(bottom, top)
    ax1.set_xticklabels(np.repeat(randomDists, 1),rotation=60, fontsize=28,horizontalalignment='center',verticalalignment='top')

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    pos = np.arange(numBoxes) + 1
    upperLabels = [str(np.round(s, 3)) for s in medians]
    weights = ['bold', 'semibold']
    for tick, label in zip(range(numBoxes), ax1.get_xticklabels()):
        k = tick % 2
        ax1.text(pos[tick], 1.005, upperLabels[tick],
                 horizontalalignment='center', fontsize=28, weight=weights[k],
                 color=boxColors[k])

    # Finally, add a basic legend
    # fig.text(0.80, 0.08, 'Training',
    #         backgroundcolor=boxColors[0], color='black', weight='roman',
    #         size='x-small')
    # fig.text(0.80, 0.045, 'Testing',
    #         backgroundcolor=boxColors[1],
    #         color='white', weight='roman', size='x-small')
    # fig.text(0.80, 0.015, '*', color='black',
    #          weight='roman', size='medium')
    # fig.text(0.815, 0.013, ' Average Value', color='black', weight='roman',
    #          size='x-small')
    fig.text(0.4, 0.36, 'T',color=boxColors[0],bbox=dict(facecolor=boxColors[0], edgecolor='black', boxstyle='round,pad=1'),weight='roman',fontsize=6)
    fig.text(0.415, 0.35, 'Training',color='black', weight='roman',fontsize=24)
    fig.text(0.53, 0.36, 'T',color=boxColors[1],bbox=dict(facecolor=boxColors[1], edgecolor='black', boxstyle='round,pad=1'),weight='roman',fontsize=6)#backgroundcolor=boxColors[1],,'Testing',color='black',
    fig.text(0.545, 0.35, 'Testing',color='black', weight='roman',fontsize=24)
    fig.text(0.65, 0.35, '*', color='black',weight='roman',  fontsize=30)
    fig.text(0.665, 0.35, ' Average Value', color='black', weight='roman',
              fontsize=26)
    accuarcy_comparison_image_save_path='compare_image_history_accuarcy/acc_'+num_of_experment
    if not os.path.exists(accuarcy_comparison_image_save_path):
        os.makedirs(accuarcy_comparison_image_save_path)
    fig.savefig(accuarcy_comparison_image_save_path+'/accuarcy_comparision.png',bbox_inches='tight')
    plt.show()