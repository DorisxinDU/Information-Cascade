#=====================================library needed===============================================================
from __future__ import print_function
import pickle as cPickle
import numpy as np
import plot as plo
import matplotlib.pyplot as plt
import os
from training_parameters import parameters # Import parametrs used for training model
#==============================================================================================
#                           function needed
#==============================================================================================
def add_subplot_axes(ax,rect):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height])
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax
def load_DATA(name):
    with open(name, 'rb') as f:
        data=cPickle.load(f)
        return data
#========================load parameters=======================
#    figure setting
#-----------------------------------------------------------------
xticks = np.linspace(0,14,6)
yticks = np.linspace(0.6,1.2,4)
x_lim=-0.1 
y_lim=0.6
axis_font=28
fig_size = (16,8)    
#======================================================================
for number_of_data in [17]:
    for num_of_train in range(1):
        _,_,_,nb_epoch,_,structure,num_of_experment,outputsize,percent_of_train,accuarcy_name,name_num_of_experment=parameters(number_of_data,num_of_train)
        accuarcy_save_path='history_accuarcy/acc_'+num_of_experment
        HISTORY=load_DATA(accuarcy_save_path+'/'+'acc.pkl')
        max_epoch=len(HISTORY['accuarcy'][0][accuarcy_name])
        epochsInds=np.linspace(1,max_epoch,max_epoch)
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
        fig_all=plt.figure(figsize=(16,10))
        axis_font=28
        axes=fig_all.add_subplot(1,2,1)
        axes.plot(hISTORY['iter_TRAIN'][0][:],label='Average:'+str(round(np.mean(hISTORY['iter_TRAIN'][0][-100:-1]),3)))
        # axes.text(0.3*len(hISTORY['iter_TRAIN'][0]),0.9,'Average:'+str(round(np.mean(hISTORY['iter_TRAIN'][0][-100:-1]),3)),fontsize=28)
        plt.legend(fontsize=axis_font-10,loc=9,ncol=1,bbox_to_anchor=(0.71,0.73))
        axes.set_xlim(-0.1)
        axes.set_ylim(0.6)
        axes.set_yticks(np.linspace(0.6,1,5))
        # plt.title('ACC_TRAIN',fontsize=axis_font+2)
        plt.xlabel('Epoch\n (a) Training accuarcy',fontsize=axis_font)
        plt.ylabel('Accuarcy',fontsize=axis_font)
        plt.xticks(rotation=45,fontsize=22)
        plt.yticks(fontsize=22)
        # =======================================================
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
        ax2 = fig_all.add_axes([xl,yl,width,height])
        ax2.plot(hISTORY['iter_TRAIN'][0][0:50],label='Train_acc')
        # plt.title('ACC_TRAIN',fontsize=axis_font-2)
        plt.xlabel('Epoch',fontsize=axis_font-2)
        plt.ylabel('Accuarcy',fontsize=axis_font-2)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        # plt.legend(fontsize=axis_font-10)
        # ===============================================================
        # plt.grid(True)
        
        axes1=fig_all.add_subplot(1,2,2)
        axes1.plot(hISTORY['iter_TEST'][0][:],label='Average:'+str(round(np.mean(hISTORY['iter_TEST'][0][-100:-1]),3)))
        # axes1.text(0.3*len(hISTORY['iter_TEST'][0]),0.9,'Average:'+str(round(np.mean(hISTORY['iter_TEST'][0][-100:-1]),3)),fontsize=28)
        # plt.grid(True)
        plt.legend(fontsize=axis_font-10,loc=9,ncol=1,bbox_to_anchor=(0.71,0.73))
        axes1.set_xlim(x_lim)
        axes1.set_ylim(0.6)
        axes1.set_yticks(np.linspace(0.6,1,5))
        # plt.title('ACC_TEST',fontsize=axis_font + 2)
        plt.xlabel('Epoch\n (b) Testing accuarcy',fontsize=axis_font)
        plt.ylabel('Accuarcy',fontsize=axis_font)
        plt.xticks(rotation=45,fontsize=22)
        plt.yticks(fontsize=22)
        # ========================================================
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
        ax1.plot(hISTORY['iter_TEST'][0][0:50],label='test_acc')
        # plt.title('ACC_TRAIN',fontsize=axis_font-2)
        plt.xlabel('Epoch',fontsize=axis_font-2)
        plt.ylabel('Accuarcy',fontsize=axis_font-2)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        # plt.legend(fontsize=axis_font-10)
        # ==========================================================
        # plt.suptitle('all in one'+num_of_experment)
        plt.show() 
        fig_all.savefig(accuarcy_image_save_path+'/accuarcy.png',bbox_inches="tight")
        fig_all.savefig('image_history_accuarcy/accuarcy'+num_of_experment+'.png',bbox_inches="tight")
        #=========================load information=========================================
        information_save_path='information_save/'+num_of_experment
        local_IXT=load_DATA(information_save_path+'/information_x.pkl')
        local_ITY=load_DATA(information_save_path+'/information_y.pkl')
        ##==============================plot all=================================
        information_figure_path='fully_information_result_saving/'+num_of_experment
        if not os.path.exists(information_figure_path):
            os.makedirs(information_figure_path)
        plo.plotfigure(np.array(local_IXT).T,np.array(local_ITY).T,epochsInds,information_figure_path)
        del local_IXT,local_ITY
        # ===============================================================================================
        local_IXT=load_DATA(information_save_path+'/KDEinformation_x.pkl')
        local_ITY=load_DATA(information_save_path+'/KDEinformation_y.pkl')
        ##==============================plot all=================================
        information_figure_path='fully_information_result_saving/KDE'+num_of_experment
        if not os.path.exists(information_figure_path):
            os.makedirs(information_figure_path)
        plo.plotfigure(np.array(local_IXT).T,np.array(local_ITY).T,epochsInds,information_figure_path)
        del local_IXT,local_ITY
         # ===============================================================================================
        local_IXT=load_DATA(information_save_path+'/Kdisinformation_x.pkl')
        local_ITY=load_DATA(information_save_path+'/Kdisinformation_y.pkl')
        ##==============================plot all=================================
        information_figure_path='fully_information_result_saving/Kdis'+num_of_experment
        if not os.path.exists(information_figure_path):
            os.makedirs(information_figure_path)
        plo.plotfigure(np.array(local_IXT).T,np.array(local_ITY).T,epochsInds,information_figure_path)
        del local_IXT,local_ITY
        # plo.plotsonap(local_IXT,local_ITY,xticks,yticks,x_lim,y_lim,epochsInds,information_figure_path)
#        plo.drawgif(information_figure_path,len(epochsInds),epochsInds)
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
        
        
