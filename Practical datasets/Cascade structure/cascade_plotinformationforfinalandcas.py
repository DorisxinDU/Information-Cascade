# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 14:22:36 2018

@author: xd3y15
"""


from __future__ import print_function
import pickle as cPickle
import numpy as np
import plot as plo
import matplotlib.pyplot as plt
import os

def load_DATA(name):
    with open(name, 'rb') as f:
        data=cPickle.load(f)
        return data
def parameters(number_of_data,num_of_train):
      "the structure can alter, e.g:list(map(int,np.linspace(2,100,20))) or [100,50,30,20,15,10]"
#      input_dimension=12
      percent_of_train=np.array([70.00])
      structure=[20,10,7,3]#[20,10,7,3]# The structure  of the network (The number of nodes in each layer)
      outputsize=1          # The dimension of the prediction layer
      num_of_experment='check_weight_no_stop'+str(structure)+'RULar_INFORMATION'+str(percent_of_train)+'_'+'experment_'+str(number_of_data)+'_'+str(num_of_train)#'test_INFORMATION_version_'+str(percent_of_train)+'_'+str(structure)+'_full_size_experment_'+str(number_of_data)+'_'+str(num_of_train)#str(structure)+'RULar_INFORMATION'+str(percent_of_train)+'_'+'experment_'+str(number_of_data)+'_'+str(num_of_train)#for change the saving name of results#'structure_test_cascade_learning'+str(percent_of_train)+'_'+str(structure)+'_'+'experment_'+str(number_of_data)+'_'+str(num_of_train)#'test_KDE_save_cascade_learning'+str(percent_of_train)+'_'+str(structure)+'_experment_'+str(number_of_data)+'_'+str(num_of_train)   #str(structure)+'RULar_INFORMATION'+str(percent_of_train)+'_'+'experment_'+str(number_of_data)+'_'+str(num_of_train)#for change the saving name of results
      name_num_of_experment=str(structure)+'_experment_'+str(number_of_data)
      outputsize=1 # The size of the output
      nb_epoch=3000 # The number of training epoch
      num_of_bins=25
      bins = np.linspace(-1, 1, num_of_bins)
      bins = bins.astype(np.float32)
      accuarcy_name='acc'
      return nb_epoch,structure,num_of_experment,outputsize,percent_of_train,name_num_of_experment,bins,accuarcy_name
#========================load parameters=======================
#    figure setting
#-----------------------------------------------------------------
xticks = np.linspace(0,10,11)
yticks = np.linspace(0,0.5,6)
x_lim=-0
y_lim=-0
axis_font=25
fig_size = (16,8)
font_size = 25    
#======================================================================
def plot_layer_sepreate(local_IXT_CAS,local_ITY_CAS,finallocal_IXT_CAS,finallocal_ITY_CAS,structure,num_of_experment):
        fig1=plt.figure(figsize=(15, 8))
        local_IXT_CAS_array,local_ITY_CAS_array=np.array(local_IXT_CAS),np.array(local_ITY_CAS)
        finallocal_IXT_CAS_array,finallocal_ITY_CAS_array=np.array(finallocal_IXT_CAS),np.array(finallocal_ITY_CAS)
        for layernum in range(len(structure)):
            ax=fig1.add_subplot(2,np.ceil(len(structure)/2),layernum+1)
            ax.scatter(local_IXT_CAS_array[layernum],local_ITY_CAS_array[layernum],label='CAS_layer'+str(layernum))
            print('the epoch is:'+str(len(local_IXT_CAS_array[layernum])))
            ax.scatter(local_IXT_CAS_array[layernum][-1],local_ITY_CAS_array[layernum][-1],label='CAS_layer'+str(layernum)+'final_epoch',facecolors='none', edgecolors='b')
            ax.scatter(finallocal_IXT_CAS_array[layernum],finallocal_ITY_CAS_array[layernum],label='finalCAS_layer'+str(layernum)) 
            ax.scatter(finallocal_IXT_CAS_array[layernum][-1],finallocal_ITY_CAS_array[layernum][-1],label='finalCAS_layer'+str(layernum)+'final_epoch',facecolors='none', edgecolors='g') 
            fig1.suptitle('information_comparison_'+num_of_experment,fontsize=axis_font + 2)
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            plt.xlabel('$I(X;T)$',fontsize=axis_font)
            plt.ylabel('$I(Y;T)$',fontsize=axis_font)
            plt.legend()
        # plt.show()
        information_figure_path='Cascade_information_result_saving/'+num_of_experment
        finalinformation_figure_path='Cascade_final_information_result_saving/'+num_of_experment
        if not os.path.exists(information_figure_path):
            os.makedirs(information_figure_path)
        if not os.path.exists(finalinformation_figure_path):
            os.makedirs(finalinformation_figure_path)
        fig1.savefig(information_figure_path+'/'+'hidden_final_compare.png')
#========================load=======================     
for number_of_data in ['dorothea']:#Epileptic,gisette
    for num_of_train in range(1):
        nb_epoch,structure,num_of_experment,outputsize,percent_of_train,name_num_of_experment,bins,accuarcy_name=parameters(number_of_data,num_of_train)
        
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
        axis_font=28
        axes=fig_all.add_subplot(1,2,1)
        for layernum in range(len(structure)):
            axes.plot(hISTORY['iter_TRAIN'+str(layernum)][0],label='Layer '+str(layernum)+', Average:'+str(str(round(np.mean(hISTORY['iter_TRAIN'+str(layernum)][0][-100:-1]),3))))
            # axes.text(200,0.9+0.02*layernum,'Layer '+str(layernum)+':'+str(round(np.mean(hISTORY['iter_TRAIN'+str(layernum)][0][-100:-1]),3)),fontsize=axis_font-6)#0.5*len(hISTORY['iter_TRAIN'+str(layernum)][0])
        plt.legend(fontsize=axis_font-10,loc=1,ncol=1,bbox_to_anchor=(0.98,0.75))
        # plt.grid(True)
        axes.set_xlim(-0.1)
        axes.set_ylim(0.4)
        axes.set_yticks(np.linspace(0.4,1.1,7)) 
        # plt.title('ACC_TRAIN',fontsize=axis_font+2)
        plt.xlabel('Epoch\n (a) Training accuarcy',fontsize=axis_font)
        plt.ylabel('Accuarcy',fontsize=axis_font)
        plt.xticks(rotation=45,fontsize=22)
        plt.yticks(fontsize=22)
        # ===================================================================
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
        plt.ylabel('Accuarcy',fontsize=axis_font-2)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.legend(fontsize=axis_font-10,loc='3')
        # ======================================================================
        axes1=fig_all.add_subplot(1,2,2) 
        for layernum in range(len(structure)):
            axes1.plot(hISTORY['iter_TEST'+str(layernum)][0],label='Layer '+str(layernum)+', Average:'+str(str(round(np.mean(hISTORY['iter_TEST'+str(layernum)][0][-100:-1]),3))))
            # axes1.text(200,0.9+0.02*layernum,'Layer '+str(layernum)+':'+str(round(np.mean(hISTORY['iter_TEST'+str(layernum)][0][-100:-1]),3)),fontsize=axis_font-6)
        plt.legend(fontsize=axis_font-10,loc=1,ncol=1,bbox_to_anchor=(0.98,0.75))
        # plt.grid(True)
        axes1.set_xlim(-0.1)
        axes1.set_ylim(0.4)
        axes1.set_yticks(np.linspace(0.4,1.1,7))
        # plt.title('ACC_TEST',fontsize=axis_font+2)
        plt.xlabel('Epoch\n (b) Testing accuarcy',fontsize=axis_font)
        # plt.ylabel('Accuarcy',fontsize=axis_font)
        plt.xticks(rotation=45,fontsize=22)
        plt.yticks(fontsize=22)
        # ===========================================================================
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
        plt.ylabel('Accuarcy',fontsize=axis_font-2)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.legend(fontsize=axis_font-10,loc='3')
        plt.show()
        # ==========================================================================
        # plt.suptitle('all in one'+num_of_experment)
        #plt.show()
        fig_all.savefig(accuarcy_image_save_path+'/accuarcy.png',bbox_inches='tight')
        fig_all.savefig('image_history_accuarcy/accuarcy'+num_of_experment+'.png',bbox_inches='tight')
        ####==========================cas================================================
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
        plo.plotfigure(local_IXT_CAS_array,local_ITY_CAS_array,epochsInds,information_figure_path)
        plo.plotfigure(finallocal_IXT_CAS_array,finallocal_ITY_CAS_array,epochsInds,finalinformation_figure_path)
        # plo.finalplotfigure(finallocal_IXT_CAS_array.T,finallocal_ITY_CAS_array.T,local_IXT_CAS_array.T,local_ITY_CAS_array.T,epochsInds,information_figure_path)
        del local_IXT_CAS,local_ITY_CAS,finallocal_IXT_CAS,finallocal_ITY_CAS,information_figure_path,finalinformation_figure_path,local_IXT_CAS_array,local_ITY_CAS_array,finallocal_IXT_CAS_array,finallocal_ITY_CAS_array
        # ===============================================================================
        # information_save_path='cascade_information_save/'+num_of_experment
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
        # plo.finalplotfigure(finallocal_IXT_CAS_array.T,finallocal_ITY_CAS_array.T,local_IXT_CAS_array.T,local_ITY_CAS_array.T,epochsInds,information_figure_path)
        del local_IXT_CAS,local_ITY_CAS,finallocal_IXT_CAS,finallocal_ITY_CAS,information_figure_path,finalinformation_figure_path,local_IXT_CAS_array,local_ITY_CAS_array,finallocal_IXT_CAS_array,finallocal_ITY_CAS_array
        #====================================================================================
        # information_save_path='cascade_information_save/'+num_of_experment
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
        # plo.finalplotfigure(finallocal_IXT_CAS_array.T,finallocal_ITY_CAS_array.T,local_IXT_CAS_array.T,local_ITY_CAS_array.T,epochsInds,information_figure_path)
        del local_IXT_CAS,local_ITY_CAS,finallocal_IXT_CAS,finallocal_ITY_CAS,information_figure_path,finalinformation_figure_path,local_IXT_CAS_array,local_ITY_CAS_array,finallocal_IXT_CAS_array,finallocal_ITY_CAS_array
        # ===============================================================================
        #plo.plotsonap(local_IXT,local_ITY,xticks,yticks,x_lim,y_lim,epochsInds,save_name)
        #plo.plotsonap(local_IXT_CAS,local_ITY_CAS,xticks,yticks,x_lim,y_lim,epochsInds,save_name_cas)
        ###==============================================================
        ##fig=plo.plot3D(local_IXT,local_ITY,'epoch_lay'+storename)
        ##fig1=plo.plot3D(local_IXT_CAS,local_ITY_CAS,'epoch_lay_cas'+storename)
        #plo.drawgif(save_name,len(epochsInds),epochsInds)
        #plo.drawgif(save_name_cas,len(epochsInds),epochsInds)
        ###=============================================================================
        ####=======================================================
        ##def display_gif(fn):
        ##    from IPython import display
        ##    return display.HTML('<img src="{}">'.format(fn))  
        ##display_gif(save_name+'created_gif.gif')
        ##display_gif(save_name_cas+'created_gif.gif')
        ###==========================================================================
        ##fig=plo.plot3D(local_IXT,local_ITY,'3Depoch_lay'+storename)
        ##fig1=plo.plot3D(local_IXT_CAS,local_ITY_CAS,'3Depoch_lay_cas'+storename)
        ##from matplotlib import animation
        ##from mpl_toolkits.mplot3d import Axes3D
        ##def init():
        ##    fig=plo.plot3D(local_IXT,local_ITY,'epoch_lay'+storename)
        ##    return fig,
        ##  
        ##fig=plo.plot3D(local_IXT,local_ITY,'epoch_lay'+storename)
        #fig = plt.figure()
        #ax = Axes3D(fig)
        #def animate(i):
        ##    fig.axes[0].view_init(elev=10., azim=i)
        #    ax.view_init(elev=10., azim=i)
        #    return fig,
        #  
        #anim = animation.FuncAnimation(fig, animate, init_func=None,
        #                               frames=360, interval=20, blit=True)
        ## Save
        #anim.save("11.mp4", writer='ffmpeg')
        ##def animate(i):
        ##    fig1.axes[0].view_init(elev=10., azim=i)
        ##    return fig
        ##  
        ##anim = animation.FuncAnimation(fig1, animate, init_func=None,
        ##                               frames=360, interval=20, blit=True)
        ### Save
        ##anim.save('epoch_lay_cas'+storename+".mp4", writer='ffmpeg')
        
        
