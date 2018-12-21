# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 15:44:10 2018

@author: xd3y15
"""
#import seaborn as sns
#sns.set_style('darkgrid')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
#import matplotlib
#matplotlib.use('Agg')
#matplotlib.use("TkAgg")
import numpy as np
import _pickle as cPickle
# import cPickle
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import scipy.io as sio
import scipy.stats as sis
import os
import matplotlib.animation as animation
import math
import os.path
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tkinter as tk
from numpy import linalg as LA

from tkinter import filedialog
LAYERS_COLORS  = ['red', 'blue', 'green', 'yellow', 'pink', 'orange']
def load_figures(title):
    """Creaet new figure based on the mode of it
    This function is really messy and need to rewrite """
    axis_font=28
    bar_font=28
#    fig_size = (12,6)
    font_size=28
    f, axes=plt.subplots(figsize = (16,8),sharey=True)
#    f.subplots_adjust(left=0.097, bottom=0.12, right=0.87, top=0.98, wspace=0, hspace=0)
#    f.set_size_inches(8.4, 6)
    colorbar_axis = np.array([0.9, 0.12, 0.03, 0.8])
#    xticks = np.linspace(0,15,6)
##    yticks = [0, 0.2, 0.4, 0.6, 0.8, 1,1.2,1.4,2,2.5,3,3.5]
#    yticks = np.linspace(0,1,5)
    xticks = np.linspace(0,14,6)
#    yticks = [0, 0.2, 0.4, 0.6, 0.8, 1,1.2,1.4,2,2.5,3,3.5]
    yticks = np.linspace(0,1.2,5)
#    xticks = np.linspace(-0.2,16,10)
#    yticks = np.linspace(-0.2,1.2,7)
    sizes = [[-1]]
    title_strs = [['', '']]
    plt.title(title,fontsize=axis_font+2)
    return font_size, axis_font, bar_font, colorbar_axis, sizes, yticks, xticks,title_strs, f, axes
def create_color_bar(f, cmap, colorbar_axis, bar_font, epochsInds, title):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    cbar_ax = f.add_axes(colorbar_axis)
    cbar = f.colorbar(sm, ticks=[], cax=cbar_ax)
    cbar.ax.tick_params(labelsize=bar_font)
    cbar.set_label(title, size=bar_font)
    cbar.ax.text(0.5, -0.01, epochsInds[0], transform=cbar.ax.transAxes,
                 va='top', ha='center', size=bar_font)
    cbar.ax.text(0.5, 1.0, str(epochsInds[-1]), transform=cbar.ax.transAxes,
                 va='bottom', ha='center', size=bar_font)
def adjustAxes(axes, axis_font=28, title_str='', x_ticks=[], y_ticks=[], x_lim=None, y_lim=None,
               set_xlabel=True, set_ylabel=True, x_label='', y_label='', set_xlim=True, set_ylim=True, set_ticks=True,
               label_size=28, set_yscale=False,
               set_xscale=False, yscale=None, xscale=None, ytick_labels='', genreal_scaling=False):
    """Organize the axes of the given figure"""
    if set_xscale:
        axes.set_xscale(xscale)
    if set_yscale:
        axes.set_yscale(yscale)
    if genreal_scaling:
        axes.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axes.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axes.xaxis.major.formatter._useMathText = True
        axes.set_yticklabels(ytick_labels)
        axes.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
    if set_xlim:
        axes.set_xlim(-0.1)
    if set_ylim:
        axes.set_ylim(-0.1)
    axes.set_title(title_str, fontsize=axis_font + 2)
    axes.tick_params(axis='y', labelsize=axis_font)
    axes.tick_params(axis='x', labelsize=axis_font)
    if set_ticks:
        axes.set_xticks(x_ticks)
        axes.set_yticks(y_ticks)
    if set_xlabel:
        axes.set_xlabel(x_label, fontsize=label_size)
    if set_ylabel:
        axes.set_ylabel(y_label, fontsize=label_size)
def plotfigure(ix_epoch,iy_epoch,epochsInds,title):
    [font_size, axis_font, bar_font, colorbar_axis, sizes, yticks, xticks,title_strs, f, axes] = load_figures(title)
    I_XT_array=np.array(ix_epoch)
#    print('the shape of the hidden layers output is:',I_XT_array)
    I_TY_array=np.array(iy_epoch)
#    print('the shape of the hidden layers output is:',I_TY_array)
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1,len(epochsInds))]
    print('the length of color is'+str(len(colors)))
    for layer_number in range (0,I_XT_array.shape[0]): 
        print('this is layer '+str(layer_number))
        I_XT_array_1=I_XT_array[layer_number]
        I_TY_array_1=I_TY_array[layer_number]
        for index_in_range in range(0,len(I_XT_array_1)):
            # print('current index is :'+str(index_in_range))
            # print('the length of color is'+str(len(colors)))
            XT = I_XT_array_1[index_in_range]
            TY = I_TY_array_1[index_in_range]
            axes.plot(XT, TY, marker='o', linestyle='-', markersize=12,markeredgewidth=0.01,linewidth=0.2,color=colors[index_in_range])
    divider = make_axes_locatable(axes)
    colorbar_axis = divider.append_axes('right', size='5%', pad='5%')
    create_color_bar(f, cmap, colorbar_axis, bar_font, np.array(epochsInds), title='Epochs')
    adjustAxes(axes, axis_font=axis_font, title_str='', x_ticks=xticks,
                         y_ticks=yticks, x_lim=[0, 25.1], y_lim=None,
                         set_xlabel=True, set_ylabel=True, x_label='$I(X;T)$',
                         y_label='$I(T;Y)$', set_xlim=True,
                         set_ylim=True, set_ticks=True, label_size=font_size)
    plt.suptitle(title)

    f.savefig(title+'/information.png',bbox_inches="tight")
    plt.show()
def finalplotfigure(finalix_epoch,finaliy_epoch,ix_epoch,iy_epoch,epochsInds,title):
#    [font_size, axis_font, bar_font, colorbar_axis, sizes, yticks, xticks,title_strs, f1, axes] = load_figures(title)
    I_XT_array=np.array(ix_epoch)
    I_TY_array=np.array(iy_epoch)
    finalI_XT_array=np.array(finalix_epoch)
    finalI_TY_array=np.array(finaliy_epoch)
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1,len(epochsInds))]
    x_lim=-0.1
    y_lim=-0.1
    xticks = np.linspace(0,14,6)
    yticks = np.linspace(0,1.2,5)
    axis_font=28
    bar_font =28
#    colorbar_axis = np.array([0.9, 0.12, 0.94, 0.8])

    figure = plt.figure()

#    [font_size, axis_font, bar_font, colorbar_axis, sizes, yticks, xticks,title_strs, figure, _] = load_figures(title)
    figure.suptitle(title+'all final',fontsize=axis_font)
    for layer_number in range (0,I_XT_array.shape[0]):
        axes=figure.add_subplot(2,np.ceil(I_XT_array.shape[0]/2),layer_number+1)
#        [font_size, axis_font, bar_font, colorbar_axis, sizes, yticks, xticks,title_strs, figure, axes] = load_figures(title)
        finalXT = finalI_XT_array[:][layer_number]
        finalTY = finalI_TY_array[:][layer_number]
        XTs = np.array([I_XT_array[:][layer_number],finalXT]).T
        TYs = np.array([I_TY_array[:][layer_number],finalTY]).T
#        print(XT.shape())
        for index_in_range in range(0,len(finalXT)):
            XT = XTs[index_in_range, :]
            TY = TYs[index_in_range, :]
            axes.plot(XT[:], TY[:], marker='o', linestyle='-', markersize=12,markeredgewidth=0.01,linewidth=0.2,color=colors[index_in_range])
#            create_color_bar(figure, cmap, colorbar_axis, bar_font, np.array(epochsInds), title='Epochs')
            axes.set_xticks(xticks)
            axes.set_yticks(yticks)
            axes.set_xlim(x_lim)
            axes.set_ylim(y_lim)
            axes.set_xlabel('$I(X;T)$',fontsize=axis_font)
            axes.set_ylabel('$I(Y;T)$',fontsize=axis_font)
            divider = make_axes_locatable(axes)
            colorbar_axis = divider.append_axes('right', size='5%', pad='5%')
            create_color_bar(figure, cmap, colorbar_axis, bar_font, np.array(epochsInds), title='Epochs')


#            plt.legend()
#        adjustAxes(axes, axis_font=axis_font, title_str='', x_ticks=xticks,
#                             y_ticks=yticks, x_lim=[0, 25.1], y_lim=None,
#                             set_xlabel=True, set_ylabel=True, x_label='$I(X;T)$',
#                             y_label='$I(T;Y)$', set_xlim=False,
#                             set_ylim=False, set_ticks=True, label_size=font_size)

#    create_color_bar(figure, cmap, colorbar_axis, bar_font, np.array(epochsInds), title='Epochs')
#    plt.tight_layout()
    figure.savefig(title+'/all final'+'.png')
    #plt.show()
#    plt.show()
def ploterr(trainacc,testacc):
    [font_size, axis_font, bar_font, colorbar_axis, sizes, _, _,title_strs, f, axes] = load_figures()
    axes.plot(trainacc,'r',label='train accuarcy')
    axes.plot(testacc,'b',label='test accuarcy')
    plt.legend()
    plt.show()
def plotsonap(local_IXT,local_ITY,xticks,yticks,x_lim,y_lim,epochsInds,save_name):
    import os.path
    LAYERS_COLORS  = ['red', 'blue', 'green', 'yellow', 'pink', 'orange','black','purple']
    colors = LAYERS_COLORS
    fig, ax = plt.subplots()
    fig.set_size_inches(6.4, 6)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    plt.xlabel('$I(X;T)$',fontsize=28)
    plt.ylabel('$I(Y;T)$',fontsize=28)
    plt.legend()
#    plt.xlabel('I_XT',fontsize=28)
#    plt.ylabel('I_YT',fontsize=28)
    ix_epoch,iy_epoch=local_IXT,local_ITY
    I_XT_array=np.array(ix_epoch)
    I_TY_array=np.array(iy_epoch)
#    labels=['Layer'+str(layer_num) for layer_num in range(I_XT_array.shape[1])]
    plt.legend()
#if num==0:
#        for layer_num in range(I_XT_array.shape[1]):
#            plt.scatter(I_XT_array[num,layer_num],I_TY_array[num,layer_num], color = colors[layer_num], s = 35,edgecolors = 'black',alpha = 0.85,label='layer'+str(layer_num))
#      else:
    for num in range (len(local_IXT)):
      fig.suptitle('Information Plane - Epoch number - ' + str(epochsInds[num]))
      if num==0:
        for layer_num in range(I_XT_array.shape[1]):
  #           points=plt.scatter(I_XT_array[num,layer_num],I_TY_array[num,layer_num], color = colors[layer_num], s = 35,edgecolors = 'black',alpha = 0.85,label="layer {}".format(layer_num))
             plt.scatter(I_XT_array[num,layer_num],I_TY_array[num,layer_num], color = colors[layer_num], s = 35,edgecolors = 'black',alpha = 0.85,label='layer'+str(layer_num))
             plt.legend()
        if not os.path.exists('./images'+'./Information Plane - Epoch number - ' + str(epochsInds[num])+save_name):
              os.makedirs('./images'+'./Information Plane - Epoch number - ' + str(epochsInds[num])+save_name)
        fig.savefig(os.path.join('./images',os.path.basename('./Information Plane - Epoch number - ' + str(epochsInds[num])+save_name+'/snap.png')))
      else:
        for layer_num in range(I_XT_array.shape[1]):
#           points=plt.scatter(I_XT_array[num,layer_num],I_TY_array[num,layer_num], color = colors[layer_num], s = 35,edgecolors = 'black',alpha = 0.85,label="layer {}".format(layer_num))
           plt.scatter(I_XT_array[num,layer_num],I_TY_array[num,layer_num], color = colors[layer_num], s = 35,edgecolors = 'black',alpha = 0.85)

#      labels=['Layer'+str(layer_num) for layer_num in range(I_XT_array.shape[1])]
#      plt.legend()
      if not os.path.exists('./images'+'Information Plane - Epoch number - ' + str(epochsInds[num])+save_name):
            os.makedirs('./images'+'Information Plane - Epoch number - ' + str(epochsInds[num])+save_name)
      fig.savefig(os.path.join('./images',os.path.basename('./images'+'Information Plane - Epoch number - ' + str(epochsInds[num])+save_name+'/snap.png')))
import imageio
def create_gif(image_list, gif_name):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread('./images'+image_name))
    # Save them as frames into a gif
    imageio.mimsave(gif_name, frames, 'GIF', duration = 0.1)
    return
def drawgif(save_name,np_epoch,epochsInds):
    image_list=[str('/images'+'Information Plane - Epoch number - ' + str(epochsInds[num])+save_name+'/snap.png')  for num in range(np_epoch)]
    gif_name = save_name +'created_gif.gif'
    create_gif(image_list, gif_name)
def plot3D(local_IXT,local_ITY,title):
    font_size = 12
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle('information_comparison_'+title,fontsize=font_size + 2)
    ax.set_xlabel('I_XT',fontsize=font_size)
    ax.set_ylabel('I_TY',fontsize=font_size)
    ax.set_zlabel('layer_number',fontsize=font_size)
    #ax.view_init(20,180)
    local_ITY=np.array(local_ITY)
    local_IXT=np.array(local_IXT)
    values = np.linspace(0,local_IXT.shape[0],local_IXT.shape[0])
    for layer in range(local_IXT.shape[1]):
        x= local_ITY[:,layer]
        y= local_IXT[:,layer]
        zs =layer+1
        p = ax.scatter3D(x, y, zs=zs, c=values, cmap='hot')
        ax.scatter(x,y,0)
    n=fig.colorbar(p, ax=ax,shrink=1.0)
    n.set_label('epoch',fontsize=font_size)
    fig.savefig(title+'.png')
    plt.show()

    #for angle in range(0, 360):
    #    ax.view_init(30, angle)
    #    plt.draw()
    #    plt.pause(.001)
    #imageio.mimsave('la', fig, 'GIF')

#def plotfigure1(ix_epoch,iy_epoch,epochsInds,title):
#    [font_size, axis_font, bar_font, colorbar_axis, sizes, yticks, xticks,title_strs, f, axes] = load_figures(title)
#    I_XT_array=np.array(ix_epoch).T
#    I_TY_array=np.array(iy_epoch).T
#    cmap = plt.get_cmap('gnuplot')
#    colors = [cmap(i) for i in np.linspace(0, 1,I_XT_array.shape[0])]
#    for index_in_range in range(0,I_XT_array.shape[0]-1):
#        XT = I_XT_array[index_in_range]
#        TY = I_TY_array[index_in_range]
#        axes.plot(XT, TY, marker='o', linestyle='-', markersize=12,markeredgewidth=0.01,linewidth=0.2,color=colors[index_in_range])
#        create_color_bar(f, cmap, colorbar_axis, bar_font, np.array(epochsInds), title='Epochs')
#    adjustAxes(axes, axis_font=axis_font, title_str='', x_ticks=xticks,
#                         y_ticks=yticks, x_lim=[0, 25.1], y_lim=None,
#                         set_xlabel=True, set_ylabel=True, x_label='$I(X;T)$',
#                         y_label='$I(T;Y)$', set_xlim=False,
#                         set_ylim=True, set_ticks=False, label_size=font_size)
#    plt.show()
#def plot_animation(save_name,epochsInds,local_IXT_CAS_array,local_ITY_CAS_array):
#    """Plot the movie for all the networks in the information plane"""
#    epochs_bins = [0, 500, 1500, 3000, 6000, 10000, 20000]
#    f, (axes) = plt.subplots(1, 1)
#    f.subplots_adjust(left=0.14, bottom=0.1, right=.928, top=0.94, wspace=0.13, hspace=0.55)
#    colors = LAYERS_COLORS
#    #new/old version
#    Ix = local_IXT_CAS_array
#    Iy = local_ITY_CAS_array
#    #Interploation of the samplings (because we don't cauclaute the infomration in each epoch)
#    interp_data_x = interp1d(epochsInds,Ix, axis=1)
#    interp_data_y = interp1d(epochsInds,Iy, axis=1)
#    new_x = np.arange(0,epochsInds[-1])
#    new_data = np.array([interp_data_x(new_x), interp_data_y(new_x)])
#    """"
#    train_data = interp1d(epochsInds,  np.squeeze(train_data), axis=1)(new_x)
#    test_data = interp1d(epochsInds,  np.squeeze(test_data), axis=1)(new_x)
#    """
#    line_ani = animation.FuncAnimation(f, update_line, len(new_x), repeat=False,
#                                       interval=1, blit=False, fargs=(new_data, axes,new_x,epochs_bins,colors,epochsInds))
#    Writer = animation.writers['ffmpeg']
#    writer = Writer(fps=100)
#    #Save the movie
#    line_ani.savefig(save_name+'_movie2.mp4',writer=writer,dpi=250)
##    plt.show()
#def update_line(num,new_data, axes,new_x,epochs_bins,colors,epochsInds,
#                font_size = 18, axis_font=16, x_lim = [0,12.2], y_lim=[0, 1.08], x_ticks = [], y_ticks = []):
#    """Update the figure of the infomration plane for the movie"""
#    #Print the line between the points
##    cmap=ListedColormap(LAYERS_COLORS)
#    axes[0].clear()
#    #Print the points
#    for layer_num in range(new_data[0].shape[1]):
#        axes[0].scatter(new_data[0][num,layer_num], new_data[1][num,layer_num], color = colors[layer_num], s = 35,edgecolors = 'black',alpha = 0.85)
#
#
#    title_str = 'Information Plane - Epoch number - ' + str(epochsInds[num])
#    adjustAxes(axes[0], axis_font, title_str, x_ticks, y_ticks, x_lim, y_lim, set_xlabel=True, set_ylabel=True,
#                     x_label='$I(X;T)$', y_label='$I(T;Y)$')
#


#    ax = plt.figure().add_subplot(111,projection='3d')
#ax.scatter(x, y, z,c='r',marker='o')
#for i in range(len(x)):
#    ax.text(x[i],y[i],z[i],i)

