from __future__ import print_function
"The functions of ploting results and information plane"
import pickle as cPickle
import numpy as np
import plot as plo
import matplotlib.pyplot as plt
import os
from training_parameters import parameters # Import parametrs used for training model
# ============================================================================================================
def load_DATA(name):
    "Load saved data"
    with open(name, 'rb') as f:
        data=cPickle.load(f)
        return data  
#---------------------figure setting--------------------------------------------
xticks = np.linspace(0,14,6)
yticks = np.linspace(0,1.2,5)
x_lim=-0.1 
y_lim=-0.1
axis_font=28
fig_size = (16,8)   
#======================================================================
def plot_layer_sepreate(local_IXT_CAS,local_ITY_CAS,finallocal_IXT_CAS,finallocal_ITY_CAS,structure,num_of_experment):
        "Plot information separatly for each layer and corresponding classifier"
        fig1=plt.figure(figsize=fig_size)
        local_IXT_CAS_array,local_ITY_CAS_array=np.array(local_IXT_CAS),np.array(local_ITY_CAS)
        finallocal_IXT_CAS_array,finallocal_ITY_CAS_array=np.array(finallocal_IXT_CAS),np.array(finallocal_ITY_CAS)
        for layernum in range(len(structure)):
            ax=fig1.add_subplot(2,np.ceil(len(structure)/2),layernum+1)
            ax.scatter(local_IXT_CAS_array[layernum],local_ITY_CAS_array[layernum],label='CAS_layer'+str(layernum))
            print('the epoch is:'+str(len(local_IXT_CAS_array[layernum])))
            ax.scatter(local_IXT_CAS_array[layernum][-1],local_ITY_CAS_array[layernum][-1],label='CAS_layer'+str(layernum)+'final_epoch',facecolors='none', edgecolors='b')
            ax.scatter(finallocal_IXT_CAS_array[layernum],finallocal_ITY_CAS_array[layernum],label='finalCAS_layer'+str(layernum)) 
            ax.scatter(finallocal_IXT_CAS_array[layernum][-1],finallocal_ITY_CAS_array[layernum][-1],label='finalCAS_layer'+str(layernum)+'final_epoch',facecolors='none', edgecolors='g') 
            fig1.suptitle('information_comparison_'+num_of_experment,fontsize=axis_font)
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            plt.xlabel('$I(X;T)$',fontsize=axis_font)
            plt.ylabel('$I(Y;T)$',fontsize=axis_font)
            plt.legend()
        # plt.show() # Show the picture after ploting
        information_figure_path='Cascade_information_result_saving/'+num_of_experment
        finalinformation_figure_path='Cascade_final_information_result_saving/'+num_of_experment
        if not os.path.exists(information_figure_path):
            os.makedirs(information_figure_path)
        if not os.path.exists(finalinformation_figure_path):
            os.makedirs(finalinformation_figure_path)
        fig1.savefig(information_figure_path+'/'+'hidden_final_compare.png')
#========================load=======================     
for number_of_data in [17]:
    for num_of_train in range(1):
        _,_,_,nb_epoch,_,structure,num_of_experment,outputsize,percent_of_train,accuarcy_name,name_num_of_experment=parameters(number_of_data,num_of_train)
        ###===========================plot training perfprmance========================
        accuarcy_save_path='history_accuarcy/acc_'+num_of_experment
        HISTORY=load_DATA(accuarcy_save_path+'/acc.pkl')
        max_epoch=np.array([len(HISTORY['iter'+str(layernum)][0][accuarcy_name]) for layernum in range(len(structure))]).max()
        print('The maximium epoch is:',str(max_epoch))
        epochsInds=np.linspace(1,max_epoch,max_epoch)
        hISTORY=dict()
        for layernum in range(len(structure)):
            hISTORY['iter_TRAIN'+str(layernum)]=[]
            hISTORY['iter_TEST'+str(layernum)]=[]
            for i in range(len(HISTORY['iter'+str(layernum)])):
                  hISTORY['iter_TRAIN'+str(layernum)].append(HISTORY['iter'+str(layernum)][i][accuarcy_name]) 
                  hISTORY['iter_TEST'+str(layernum)].append(HISTORY['iter'+str(layernum)][i]['val_'+accuarcy_name])
        #--------------------------------------------------------------------------------------------------------------------------------------
        accuarcy_image_save_path='image_history_accuarcy/acc_'+num_of_experment
        if not os.path.exists(accuarcy_image_save_path):
            os.makedirs(accuarcy_image_save_path)
        fig_all=plt.figure(figsize=(16,10))
        axes=fig_all.add_subplot(1,2,1)
        for layernum in range(len(structure)):
            axes.plot(hISTORY['iter_TRAIN'+str(layernum)][0],label='Layer '+str(layernum)+', Average:'+str(round(np.mean(hISTORY['iter_TRAIN'+str(layernum)][0][-100:-1]),3)))           
        plt.legend(fontsize=axis_font-10,loc=9,ncol=1,bbox_to_anchor=(0.62,0.77))
        # plt.grid(True)
        axes.set_xlim(-0.1)
        axes.set_ylim(0.6)
        axes.set_yticks(np.linspace(0.6,1,5)) 
        plt.xlabel('Epoch\n (a) Training accuarcy',fontsize=axis_font)
        plt.ylabel('Accuarcy',fontsize=axis_font)
        plt.xticks(rotation=45,fontsize=22)
        plt.yticks(fontsize=22)
        # ===========================================Plot embedding figure over first 50 epochs================================================================
        rect = [0.36,0.12,0.6,0.4]
        box = axes.get_position()
        width = box.width
        height = box.height
        inax_position  = axes.transAxes.transform(rect[0:2])
        transFigure = fig_all.transFigure.inverted()
        infig_position = transFigure.transform(inax_position)    
        xl = infig_position[0]
        yl = infig_position[1]
        width *= rect[2]
        height *= rect[3]  # <= Typo was here
        ax2= fig_all.add_axes([xl,yl,width,height])
        for layernum in range(len(structure)):
            ax2.plot(hISTORY['iter_TRAIN'+str(layernum)][0][0:50],label='Layer '+str(layernum))
        plt.xlabel('Epoch',fontsize=axis_font-2)
        ax2.set_xlim(0.0)
        plt.ylabel('Accuarcy',fontsize=axis_font-2)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.legend(fontsize=axis_font-10)
        # plt.show()
        # ===========================================================================================================
        axes1=fig_all.add_subplot(1,2,2) 
        for layernum in range(len(structure)):
            axes1.plot(hISTORY['iter_TEST'+str(layernum)][0],label='Layer '+str(layernum)+', Average:'+str(round(np.mean(hISTORY['iter_TEST'+str(layernum)][0][-100:-1]),3)))
        plt.legend(fontsize=axis_font-10,loc=1,ncol=1,bbox_to_anchor=(0.985,0.77))
        # plt.grid(True)
        axes1.set_xlim(-0.1)
        axes1.set_ylim(0.6)
        axes1.set_yticks(np.linspace(0.6,1,5))
        # plt.title('ACC_TEST',fontsize=axis_font+2)
        plt.xlabel('Epoch\n (b) Testing accuarcy',fontsize=axis_font)
        plt.ylabel('Accuarcy',fontsize=axis_font)
        plt.xticks(rotation=45,fontsize=22)
        plt.yticks(fontsize=22)
        # plt.suptitle('all in one'+num_of_experment)
        # ==========================Plot embedding figure over first 50 epochs==============================================
        rect = [0.36,0.12,0.6,0.4]
        box = axes1.get_position()
        width = box.width
        height = box.height
        inax_position  = axes1.transAxes.transform(rect[0:2])
        transFigure = fig_all.transFigure.inverted()
        infig_position = transFigure.transform(inax_position)    
        xl = infig_position[0]
        yl = infig_position[1]
        width *= rect[2]
        height *= rect[3]  # <= Typo was here
        ax1= fig_all.add_axes([xl,yl,width,height])
        for layernum in range(len(structure)):
            ax1.plot(hISTORY['iter_TEST'+str(layernum)][0][0:50],label='Layer '+str(layernum))
        plt.xlabel('Epoch',fontsize=axis_font-2)
        ax1.set_xlim(0.0)
        plt.ylabel('Accuarcy',fontsize=axis_font-2)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.legend(fontsize=axis_font-10)
        plt.show()
        fig_all.savefig(accuarcy_image_save_path+'/accuarcy.png',bbox_inches="tight")
        fig_all.savefig('image_history_accuarcy/accuarcy'+num_of_experment+'.png',bbox_inches="tight")
        ##==========================cas================================================
        information_save_path='cascade_information_save/'+num_of_experment
        local_IXT_CAS=load_DATA(information_save_path+'/information_x.pkl')
        local_ITY_CAS=load_DATA(information_save_path+'/information_y.pkl')
        finallocal_IXT_CAS=load_DATA(information_save_path+'/final_information_x.pkl')
        finallocal_ITY_CAS=load_DATA(information_save_path+'/final_information_y.pkl')
        #==============================plot final============================================
        information_figure_path='Cascade_information_result_saving/'+num_of_experment
        finalinformation_figure_path='Cascade_final_information_result_saving/'+num_of_experment
        if not os.path.exists(information_figure_path):
            os.makedirs(information_figure_path) 
        if not os.path.exists(finalinformation_figure_path):
            os.makedirs(finalinformation_figure_path)
        plot_layer_sepreate(local_IXT_CAS,local_ITY_CAS,finallocal_IXT_CAS,finallocal_ITY_CAS,structure,num_of_experment)
        local_IXT_CAS_array,local_ITY_CAS_array=np.array(local_IXT_CAS),np.array(local_ITY_CAS)
        finallocal_IXT_CAS_array,finallocal_ITY_CAS_array=np.array(finallocal_IXT_CAS),np.array(finallocal_ITY_CAS)
        print('Starting colorful map........')
        plo.plotfigure(local_IXT_CAS_array,local_ITY_CAS_array,epochsInds,information_figure_path)
        print('Starting final information')
        plo.plotfigure(finallocal_IXT_CAS_array,finallocal_ITY_CAS_array,epochsInds,finalinformation_figure_path)
        del local_IXT_CAS,local_ITY_CAS,finallocal_IXT_CAS,finallocal_ITY_CAS,information_figure_path,finalinformation_figure_path,local_IXT_CAS_array,local_ITY_CAS_array,finallocal_IXT_CAS_array,finallocal_ITY_CAS_array
        # ===============================================================================
        local_IXT_CAS=load_DATA(information_save_path+'/KDEinformation_x.pkl')
        local_ITY_CAS=load_DATA(information_save_path+'/KDEinformation_y.pkl')
        finallocal_IXT_CAS=load_DATA(information_save_path+'/KDEfinal_information_x.pkl')
        finallocal_ITY_CAS=load_DATA(information_save_path+'/KDEfinal_information_y.pkl')
        #==============================plot final============================================
        information_figure_path='Cascade_information_result_saving/KDE'+num_of_experment
        finalinformation_figure_path='Cascade_final_information_result_saving/KDE'+num_of_experment
        if not os.path.exists(information_figure_path):
            os.makedirs(information_figure_path) 
        if not os.path.exists(finalinformation_figure_path):
            os.makedirs(finalinformation_figure_path)
        plot_layer_sepreate(local_IXT_CAS,local_ITY_CAS,finallocal_IXT_CAS,finallocal_ITY_CAS,structure,'KDE'+num_of_experment)
        local_IXT_CAS_array,local_ITY_CAS_array=np.array(local_IXT_CAS),np.array(local_ITY_CAS)
        finallocal_IXT_CAS_array,finallocal_ITY_CAS_array=np.array(finallocal_IXT_CAS),np.array(finallocal_ITY_CAS)
        plo.plotfigure(local_IXT_CAS_array,local_ITY_CAS_array,epochsInds,information_figure_path)
        plo.plotfigure(finallocal_IXT_CAS_array,finallocal_ITY_CAS_array,epochsInds,finalinformation_figure_path)
        del local_IXT_CAS,local_ITY_CAS,finallocal_IXT_CAS,finallocal_ITY_CAS,information_figure_path,finalinformation_figure_path,local_IXT_CAS_array,local_ITY_CAS_array,finallocal_IXT_CAS_array,finallocal_ITY_CAS_array
        #====================================================================================
        local_IXT_CAS=load_DATA(information_save_path+'/Kdisinformation_x.pkl')
        local_ITY_CAS=load_DATA(information_save_path+'/Kdisinformation_y.pkl')
        finallocal_IXT_CAS=load_DATA(information_save_path+'/Kdisfinal_information_x.pkl')
        finallocal_ITY_CAS=load_DATA(information_save_path+'/Kdisfinal_information_y.pkl')
        #==============================plot final============================================
        information_figure_path='Cascade_information_result_saving/Kdis'+num_of_experment
        finalinformation_figure_path='Cascade_final_information_result_saving/Kdis'+num_of_experment
        if not os.path.exists(information_figure_path):
            os.makedirs(information_figure_path) 
        if not os.path.exists(finalinformation_figure_path):
            os.makedirs(finalinformation_figure_path)
        plot_layer_sepreate(local_IXT_CAS,local_ITY_CAS,finallocal_IXT_CAS,finallocal_ITY_CAS,structure,'Kdis'+num_of_experment)
        local_IXT_CAS_array,local_ITY_CAS_array=np.array(local_IXT_CAS),np.array(local_ITY_CAS)
        finallocal_IXT_CAS_array,finallocal_ITY_CAS_array=np.array(finallocal_IXT_CAS),np.array(finallocal_ITY_CAS)
        plo.plotfigure(local_IXT_CAS_array,local_ITY_CAS_array,epochsInds,information_figure_path)
        plo.plotfigure(finallocal_IXT_CAS_array,finallocal_ITY_CAS_array,epochsInds,finalinformation_figure_path)
        del local_IXT_CAS,local_ITY_CAS,finallocal_IXT_CAS,finallocal_ITY_CAS,information_figure_path,finalinformation_figure_path,local_IXT_CAS_array,local_ITY_CAS_array,finallocal_IXT_CAS_array,finallocal_ITY_CAS_array