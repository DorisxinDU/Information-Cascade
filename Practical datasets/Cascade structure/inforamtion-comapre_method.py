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
      structure=[5,3,1]# The structure  of the network (The number of nodes in each layer)
      outputsize=1          # The dimension of the prediction layer
      num_of_experment=str(structure)+'RULar_INFORMATION'+str(percent_of_train)+'_experment_'+str(number_of_data)+'_'+str(num_of_train)    #for change the saving name of results
      name_num_of_experment=str(structure)+'_experment_'+str(number_of_data)
      outputsize=1 # The size of the output
      nb_epoch=30000 # The number of training epoch
      num_of_bins=25
      bins = np.linspace(-1, 1, num_of_bins)
      bins = bins.astype(np.float32)
      accuarcy_name='acc'
      return nb_epoch,structure,num_of_experment,outputsize,percent_of_train,name_num_of_experment,bins,accuarcy_name
#========================load parameters=======================
#    figure setting
#-----------------------------------------------------------------
# xticks = np.linspace(0,14,6)
# yticks = np.linspace(0,1.2,5)
# x_lim=-0.1 
# y_lim=-0.1
axis_font=28
# fig_size = (15, 18)
font_size = 28     
#======================================================================
#========================load=======================     
for number_of_data in ['dexter']:
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
        # accuarcy_image_save_path='image_history_accuarcy/acc_'+num_of_experment
        # if not os.path.exists(accuarcy_image_save_path):
        #     os.makedirs(accuarcy_image_save_path)
        # fig_all=plt.figure(figsize=(15, 15))
        # axis_font=20
        # axes=fig_all.add_subplot(1,2,1)
        # for layernum in range(len(structure)):
        #     axes.plot(hISTORY['iter_TRAIN'+str(layernum)][0],label='CAS_train_acc_layer'+str(layernum))
        #     axes.text(2000,0.5+0.1*layernum,'Layer_'+str(layernum)+'_accuarcy is:'+str(round(hISTORY['iter_TRAIN'+str(layernum)][0][-1],4)),fontsize=15)#0.5*len(hISTORY['iter_TRAIN'+str(layernum)][0])
        # plt.legend(fontsize=axis_font)
        # plt.grid(True)
        # axes.set_xlim(-0.1)
        # axes.set_ylim(-0.1)
        # axes.set_yticks(np.linspace(0,1,11)) 
        # plt.title('ACC_TRAIN',fontsize=axis_font+2)
        # plt.xlabel('Epoch',fontsize=axis_font)
        # plt.ylabel('ACC',fontsize=axis_font)
        # plt.xticks(fontsize=15)
        # plt.yticks(fontsize=15)
        # axes=fig_all.add_subplot(1,2,2) 
        # for layernum in range(len(structure)):
        #     axes.plot(hISTORY['iter_TEST'+str(layernum)][0],label='CAS_test_acc_layer'+str(layernum))
        #     axes.text(2000,0.5+0.1*layernum,'Layer_'+str(layernum)+'_accuarcy is:'+str(round(hISTORY['iter_TEST'+str(layernum)][0][-1],4)),fontsize=15)
        # plt.legend(fontsize=axis_font)
        # plt.grid(True)
        # axes.set_xlim(-0.1)
        # axes.set_ylim(-0.1)
        # axes.set_yticks(np.linspace(0,1,11))
        # plt.title('ACC_TEST',fontsize=axis_font+2)
        # plt.xlabel('Epoch',fontsize=axis_font)
        # plt.ylabel('ACC',fontsize=axis_font)
        # plt.xticks(fontsize=15)
        # plt.yticks(fontsize=15)
        # plt.suptitle('all in one'+num_of_experment)
        # #plt.show()
        # fig_all.savefig(accuarcy_image_save_path+'/accuarcy.png')
        # fig_all.savefig('image_history_accuarcy/accuarcy'+num_of_experment+'.png')
        ####==========================cas================================================
        information_save_path='cascade_information_save/'+num_of_experment
        local_IXT_CAS=load_DATA(information_save_path+'/information_x.pkl')
        local_ITY_CAS=load_DATA(information_save_path+'/information_y.pkl')
        finallocal_IXT_CAS=load_DATA(information_save_path+'/final_information_x.pkl')
        finallocal_ITY_CAS=load_DATA(information_save_path+'/final_information_y.pkl')
#         # ===================================plot video===================================================
#         fig = plt.figure(figsize=(15,15))
#         ax1 = fig.add_subplot(2, 2, 1)
#         ax=fig.add_subplot(2, 2, 2)
#         ax2 = fig.add_subplot(2, 2, 3)
#         ax3 = fig.add_subplot(2, 2, 4)
#         acctest,acc,epoch=[],[],[]
#         values = np.linspace(0,max_epoch,max_epoch)
#         lnn=ax.scatter(local_IXT_CAS[max_index], local_ITY_CAS[max_index] ,c=values[0:max_epoch],cmap='summer')
#         lnn.remove()
#         nn=fig.colorbar(lnn,ax=ax,shrink=1.0)
#         nn.set_label('epoch',fontsize=font_size)
#         ax.set_xlabel('$I(X;T)$',fontsize=font_size)
#         ax.set_ylabel('$I(Y;T)$',fontsize=font_size)
#         for laynum in range(len(structure)):
#             ax.scatter(local_IXT_CAS[laynum], local_ITY_CAS[laynum] ,c=values[0:len(local_IXT_CAS[laynum])],cmap='summer')
#         ln=ax1.scatter(local_IXT_CAS[max_index], local_ITY_CAS[max_index] ,c=values[0:max_epoch],cmap='summer')
#         ln.remove()
#         n=fig.colorbar(ln,ax=ax1,shrink=1.0)
#         n.set_label('epoch',fontsize=font_size)
#         ax1.set_xlabel('$I(X;T)$',fontsize=font_size)
#         ax1.set_ylabel('$I(Y;T)$',fontsize=font_size)
#         ln1, =ax2.plot([], [],animated=True)
#         ax2.set_xlabel('Epoch',fontsize=font_size)
#         ax2.set_ylabel('Accuarcy',fontsize=font_size)
#         ln2, =ax3.plot([], [], animated=True)
#         ax3.set_xlabel('Epoch',fontsize=font_size)
#         ax3.set_ylabel('Accuarcy',fontsize=font_size)
#         layernum=2
#         ax2.legend(str(layernum))
#         ax3.legend(str(layernum))
#         def init():
#                 ax.set_xlim(0, 10)
#                 ax.set_ylim(0.2, 0.5)
#                 ax1.set_xlim(0, 10)
#                 ax1.set_ylim(0.2,0.5)
#                 ax2.set_xlim(0, len(local_IXT_CAS[max_index]))
#                 ax2.set_ylim(0, 1.1)
#                 ax3.set_xlim(0, len(local_IXT_CAS[max_index]))
#                 ax3.set_ylim(0, 1.1)
#                 return ln,ln1,ln2
#         def update(frame):
#                 ln=ax1.scatter(local_IXT_CAS[layernum][0:frame], local_ITY_CAS[layernum][0:frame] ,c=values[0:frame],cmap='summer')
#                 print('frame is',frame)
#                 epoch.append(frame)
#                 acc.append(hISTORY['iter_TRAIN'+str(layernum)][0][frame])
#                 acctest.append(hISTORY['iter_TEST'+str(layernum)][0][frame])
#                 ln1.set_data(epoch, acc)
#                 ln2.set_data(epoch, acctest)
                
#                 return ln,ln1,ln2
#         ani = FuncAnimation(fig, update, frames=range(0,len(local_ITY_CAS[layernum])),init_func=init,blit=True,repeat=False)
#         plt.show()
            
#         accuarcy_image_save_path='vedio_accuarcy/acc_'+num_of_experment
#         if not os.path.exists(accuarcy_image_save_path):
#              os.makedirs(accuarcy_image_save_path)
# #        ani.save(accuarcy_image_save_path+'/acc_info.mp4')

        #==============================plot final============================================
        # information_figure_path='Cascade_information_result_saving/'+num_of_experment
        # finalinformation_figure_path='Cascade_final_information_result_saving/'+num_of_experment
        # if not os.path.exists(information_figure_path):
        #     os.makedirs(information_figure_path) 
        # if not os.path.exists(finalinformation_figure_path):
        #     os.makedirs(finalinformation_figure_path)
        # plot_layer_sepreate(local_IXT_CAS,local_ITY_CAS,finallocal_IXT_CAS,finallocal_ITY_CAS,structure,num_of_experment)
        # local_IXT_CAS_array,local_ITY_CAS_array=np.array(local_IXT_CAS),np.array(local_ITY_CAS)
        # finallocal_IXT_CAS_array,finallocal_ITY_CAS_array=np.array(finallocal_IXT_CAS),np.array(finallocal_ITY_CAS)
        # print('Starting colorful map........')
        # plo.plotfigure(local_IXT_CAS_array,local_ITY_CAS_array,epochsInds,information_figure_path)
        # print('Starting final information')
        # plo.plotfigure(finallocal_IXT_CAS_array,finallocal_ITY_CAS_array,epochsInds,finalinformation_figure_path)
        # # print('Starting final information layer seperate')
        # # plo.finalplotfigure(finallocal_IXT_CAS_array.T,finallocal_ITY_CAS_array.T,local_IXT_CAS_array.T,local_ITY_CAS_array.T,epochsInds,information_figure_path)

        # del local_IXT_CAS,local_ITY_CAS,finallocal_IXT_CAS,finallocal_ITY_CAS,information_figure_path,finalinformation_figure_path,local_IXT_CAS_array,local_ITY_CAS_array,finallocal_IXT_CAS_array,finallocal_ITY_CAS_array
        # ===============================================================================
        # information_save_path='cascade_information_save/'+num_of_experment
        KDElocal_IXT_CAS=load_DATA(information_save_path+'/KDEinformation_x.pkl')
        KDElocal_ITY_CAS=load_DATA(information_save_path+'/KDEinformation_y.pkl')
        KDEfinallocal_IXT_CAS=load_DATA(information_save_path+'/KDEfinal_information_x.pkl')
        KDEfinallocal_ITY_CAS=load_DATA(information_save_path+'/KDEfinal_information_y.pkl')
        #==============================plot final============================================
        color=['b', 'g', 'r', 'k','m', 'y', 'w','c']
        fig=plt.figure(figsize=(10,10))
        axes=fig.add_subplot(2,1,1)
        axes1=fig.add_subplot(2,1,2)
        # =================================================================
        from brokenaxes import brokenaxes
        from matplotlib.gridspec import GridSpec
        # sps1, sps2 = GridSpec(2,1)
        # print('sps1:',sps1)
        # axes= brokenaxes(xlims=((0, 0.6), (0.8,1)),ylims=((0, 0.6), (0.8,1)),hspace=.15,subplot_spec=sps1)

        # ============================================================================
        for i in range(len(structure)):
            axes.plot(local_IXT_CAS[i],KDElocal_IXT_CAS[i],'.',c=color[i],label='I(X:T): Layer'+str(i))
        axes.set_xlim(-0.10)
        axes.set_ylim(-0.10)
        # axes.set_xlabel('Binning method',fontsize=axis_font)
        axes.set_ylabel('PWD method',fontsize=axis_font)
        # axes.set_xticklabels(np.linspace(2.5,12.5,11),fontsize=22)
        # axes.set_yticklabels(np.linspace(2.5,12.5,11),fontsize=22)
        axes.legend(fontsize=axis_font-10,loc=4)
        axes.tick_params(axis='both', which='major', labelsize=22)
        axes.xaxis.set_ticks(np.arange(0, 10.05, 2.))
        axes.yaxis.set_ticks(np.arange(0, 10.05, 2.))
        for i in range(len(structure)):
            axes1.plot(local_ITY_CAS[i],KDElocal_ITY_CAS[i],'.',c=color[i],label='I(Y:T): Layer'+str(i))
        axes1.legend(fontsize=axis_font-10,loc=4)
        # plt.grid()
        # axes1.set_xlim(0.58)
        # axes1.set_ylim(0.58)
        # axes.set_yticks(np.linspace(0,1,11)) 
        # plt.title('information estimation comparsion',fontsize=24)
        axes1.set_xlabel('Binning method',fontsize=axis_font)
        axes1.set_ylabel('PWD method',fontsize=axis_font)
        axes1.tick_params(axis='both', which='major', labelsize=22)
        axes1.xaxis.set_ticks(np.arange(0,1.05, 0.2))
        axes1.yaxis.set_ticks(np.arange(0,1.05, 0.2))
        # axes1.set_xticklabels(np.linspace(0.6,1.5,10),fontsize=22)
        # axes1.set_yticklabels(np.linspace(0.6,1.5,10),fontsize=22)
        plt.show()

        information_compare_path='Cascade_information_comparision_saving/'+num_of_experment
        # finalinformation_figure_path='Cascade_final_information_result_saving/KDE'+num_of_experment
        if not os.path.exists(information_compare_path):
            os.makedirs(information_compare_path) 
        fig.savefig(information_compare_path+'/info_compasion.png',bbox_inches="tight")
        # if not os.path.exists(finalinformation_figure_path):
        #     os.makedirs(finalinformation_figure_path)
        # plot_layer_sepreate(local_IXT_CAS,local_ITY_CAS,finallocal_IXT_CAS,finallocal_ITY_CAS,structure,'KDE'+num_of_experment)
        # local_IXT_CAS_array,local_ITY_CAS_array=np.array(local_IXT_CAS),np.array(local_ITY_CAS)
        # finallocal_IXT_CAS_array,finallocal_ITY_CAS_array=np.array(finallocal_IXT_CAS),np.array(finallocal_ITY_CAS)
        # plo.plotfigure(local_IXT_CAS_array,local_ITY_CAS_array,epochsInds,information_figure_path)
        # plo.plotfigure(finallocal_IXT_CAS_array,finallocal_ITY_CAS_array,epochsInds,finalinformation_figure_path)
        # plo.finalplotfigure(finallocal_IXT_CAS_array.T,finallocal_ITY_CAS_array.T,local_IXT_CAS_array.T,local_ITY_CAS_array.T,epochsInds,information_figure_path)
        # del local_IXT_CAS,local_ITY_CAS,finallocal_IXT_CAS,finallocal_ITY_CAS,information_figure_path,finalinformation_figure_path,local_IXT_CAS_array,local_ITY_CAS_array,finallocal_IXT_CAS_array,finallocal_ITY_CAS_array
        # #====================================================================================
        # # information_save_path='cascade_information_save/'+num_of_experment
        # local_IXT_CAS=load_DATA(information_save_path+'/Kdisinformation_x.pkl')
        # local_ITY_CAS=load_DATA(information_save_path+'/Kdisinformation_y.pkl')
        # finallocal_IXT_CAS=load_DATA(information_save_path+'/Kdisfinal_information_x.pkl')
        # finallocal_ITY_CAS=load_DATA(information_save_path+'/Kdisfinal_information_y.pkl')

        # #==============================plot final============================================
        # information_figure_path='Cascade_information_result_saving/Kdis'+num_of_experment
        # finalinformation_figure_path='Cascade_final_information_result_saving/Kdis'+num_of_experment
        # if not os.path.exists(information_figure_path):
        #     os.makedirs(information_figure_path) 
        # if not os.path.exists(finalinformation_figure_path):
        #     os.makedirs(finalinformation_figure_path)
        # plot_layer_sepreate(local_IXT_CAS,local_ITY_CAS,finallocal_IXT_CAS,finallocal_ITY_CAS,structure,'Kdis'+num_of_experment)
        # local_IXT_CAS_array,local_ITY_CAS_array=np.array(local_IXT_CAS),np.array(local_ITY_CAS)
        # finallocal_IXT_CAS_array,finallocal_ITY_CAS_array=np.array(finallocal_IXT_CAS),np.array(finallocal_ITY_CAS)
        # plo.plotfigure(local_IXT_CAS_array,local_ITY_CAS_array,epochsInds,information_figure_path)
        # plo.plotfigure(finallocal_IXT_CAS_array,finallocal_ITY_CAS_array,epochsInds,finalinformation_figure_path)
        # # plo.finalplotfigure(finallocal_IXT_CAS_array.T,finallocal_ITY_CAS_array.T,local_IXT_CAS_array.T,local_ITY_CAS_array.T,epochsInds,information_figure_path)
        # del local_IXT_CAS,local_ITY_CAS,finallocal_IXT_CAS,finallocal_ITY_CAS,information_figure_path,finalinformation_figure_path,local_IXT_CAS_array,local_ITY_CAS_array,finallocal_IXT_CAS_array,finallocal_ITY_CAS_array
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
        
        
