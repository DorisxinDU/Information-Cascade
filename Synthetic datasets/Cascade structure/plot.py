# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 15:44:10 2018

@author: xd3y15
"""

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import _pickle as cPickle
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
    "Figure setting"
    axis_font=bar_font=font_size = 28
    f, axes= plt.subplots(figsize=(16,8))#,sharey=True
    colorbar_axis = np.array([0.9, 0.12, 0.03, 0.8])
    xticks = np.linspace(0,14,6)
    yticks = np.linspace(0,1.2,5)
    sizes = [[-1]]
    title_strs = [['', '']]
    plt.title(title,fontsize=axis_font + 2)
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
    "Plot information plane for all cascaded layers"
    [font_size, axis_font, bar_font, colorbar_axis, sizes, yticks, xticks,title_strs, f, axes] = load_figures(title)
    I_XT_array=np.array(ix_epoch)
    I_TY_array=np.array(iy_epoch)
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1,len(epochsInds))]
    print('the length of color is'+str(len(colors)))
    for layer_number in range (0,I_XT_array.shape[0]-1): 
        print('the layer number is :',layer_number)
        I_XT_array_1=I_XT_array[layer_number]
        I_TY_array_1=I_TY_array[layer_number]
        for index_in_range in range(0,len(I_XT_array_1)):
            XT = I_XT_array_1[index_in_range]
            TY = I_TY_array_1[index_in_range]
            axes.plot(XT, TY, marker='o', linestyle='-', markersize=12,markeredgewidth=0.01,linewidth=0.2,color=colors[index_in_range])
        del I_XT_array_1,I_TY_array_1 
    divider = make_axes_locatable(axes)
    colorbar_axis = divider.append_axes('right', size='5%', pad='5%')
    create_color_bar(f, cmap, colorbar_axis, bar_font, np.array(epochsInds), title='Epochs')
    adjustAxes(axes, axis_font=axis_font, title_str='', x_ticks=xticks,
                         y_ticks=yticks, x_lim=[0, 25.1], y_lim=None,
                         set_xlabel=True, set_ylabel=True, x_label='$I(X;T)$',
                         y_label='$I(T;Y)$', set_xlim=True,
                         set_ylim=True, set_ticks=True, label_size=font_size)
    plt.suptitle(title,fontsize=18)
    f.savefig(title+'/information_same_level.png',bbox_inches="tight")
    plt.show()
    return print('plot is finished') 
def finalplotfigure(finalix_epoch,finaliy_epoch,ix_epoch,iy_epoch,epochsInds,title):
    I_XT_array=np.array(ix_epoch)
    I_TY_array=np.array(iy_epoch)
    finalI_XT_array=np.array(finalix_epoch)
    finalI_TY_array=np.array(finaliy_epoch)
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1,len(epochsInds))]
    x_lim=-0.1
    y_lim=-0.1
    xticks = np.linspace(0,16,9)
    yticks = np.linspace(0,1,11)
    axis_font=28
    bar_font = 28
    figure = plt.figure(figsize=(15, 8))
    # figure.suptitle(title+'all final',fontsize=axis_font)
    for layer_number in range (0,I_XT_array.shape[0]):
        print('the layer number is :',layer_number)
        axes=figure.add_subplot(2,np.ceil(I_XT_array.shape[0]/2),layer_number+1)
        finalXT = finalI_XT_array[:][layer_number]
        finalTY = finalI_TY_array[:][layer_number]
        XTs = np.array([I_XT_array[:][layer_number],finalXT]).T
        TYs = np.array([I_TY_array[:][layer_number],finalTY]).T
        for index_in_range in range(0,len(finalXT)):
            print('current index is :'+str(index_in_range)+'/'+str(len(finalXT)))
            XT = XTs[index_in_range, :]
            TY = TYs[index_in_range, :]
            axes.plot(XT[:], TY[:], marker='o', linestyle='-', markersize=12,markeredgewidth=0.01,linewidth=0.2,color=colors[index_in_range])
            axes.set_xticks(xticks)
            axes.set_yticks(yticks)
            axes.set_xlim(x_lim)
            axes.set_ylim(y_lim)
            axes.set_xlabel('$I(X;T)$',fontsize=axis_font)
            axes.set_ylabel('$I(Y;T)$',fontsize=axis_font)
            divider = make_axes_locatable(axes)
            colorbar_axis = divider.append_axes('right', size='5%', pad='5%')
            create_color_bar(figure, cmap, colorbar_axis, bar_font, np.array(epochsInds), title='Epochs')
    figure.savefig(title+'/all final'+'.png',bbox_inches="tight")
    plt.show()
    return print('finished ploting')
#    plt.show()
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
    ix_epoch,iy_epoch=local_IXT,local_ITY
    I_XT_array=np.array(ix_epoch)
    I_TY_array=np.array(iy_epoch)
    plt.legend()
    for num in range (len(local_IXT)):
      fig.suptitle('Information Plane - Epoch number - ' + str(epochsInds[num]))
      if num==0:
        for layer_num in range(I_XT_array.shape[1]):
             plt.scatter(I_XT_array[num,layer_num],I_TY_array[num,layer_num], color = colors[layer_num], s = 35,edgecolors = 'black',alpha = 0.85,label='layer'+str(layer_num))
             plt.legend()
        if not os.path.exists('./images'+'Information Plane - Epoch number - ' + str(epochsInds[num])+save_name):
              os.makedirs('./images'+'Information Plane - Epoch number - ' + str(epochsInds[num])+save_name)
        fig.savefig(os.path.join('./images',os.path.basename('./images'+'Information Plane - Epoch number - ' + str(epochsInds[num])+save_name+'/snap.png')))
      else:
        for layer_num in range(I_XT_array.shape[1]):
           plt.scatter(I_XT_array[num,layer_num],I_TY_array[num,layer_num], color = colors[layer_num], s = 35,edgecolors = 'black',alpha = 0.85)
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