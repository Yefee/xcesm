import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import json
import os


COLOR_PATH = os.path.join(os.path.dirname(__file__))
with open(COLOR_PATH + '/colormap.json', 'r') as f:
    colors = json.load(f)

def cmap(name, bins=None):
    data = np.array(colors[name])
    data = data / np.max(data)
    cmap = ListedColormap(data, name=name)

    if isinstance(bins, int):
        cmap = cmap._resample(bins)
    return cmap

def subplots(nrow=2, ncol=2, figsize=None, ind=None, **kwarg):
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure()

    tol = nrow * ncol   
    if ind is not None:
        ind_total = range(1, tol+1)
        ind1 = list(set(ind_total) - set(ind))
        projection = ccrs.PlateCarree(**kwarg)
        ax1 = [fig.add_subplot(nrow, ncol, i, projection=projection) for i in ind]
        ax2 = [fig.add_subplot(nrow, ncol, i) for i in ind1]
        ax = ax1 + ax2
    else:
        ind_total = range(1, tol+1)
        projection = ccrs.PlateCarree(**kwarg)
        ax = [fig.add_subplot(nrow, ncol, i, projection=projection) for i in ind_total]
   
    return ax

def make_patch_spines_invisible(ax):

    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

def change_color_spines(ax, color):
    for sp in ax.spines.values():
        sp.set_color(color)
    ax.tick_params(axis='x', which='both', colors=color)
    ax.tick_params(axis='y', which='both', colors=color)
    ax.xaxis.label.set_color(color)
    ax.yaxis.label.set_color(color)


def mk_stacking_axes(num_axes, fig=None, hsapce=-0.4, color=None, ratio=None, **kwargs):

    fig, axs = plt.subplots(num_axes, 1, sharex=True, gridspec_kw={'height_ratios':ratio}, **kwargs)

    # generate axes
    # axs = []
    # for i in range(num_axes):
    #     if i == 0:
    #         axs.append(fig.add_subplot(num_axes, 1, i+1,))
    #     else:
    #         axs.append(fig.add_subplot(num_axes, 1, i+1, sharex=axs[0]))
    
    # if ratio is not None:
    #     for i in range(num_axes):
    #         ax = axs[i]
    #         xmin, xmax = ax.get_xlim()
    #         ymin, ymax = ax.get_ylim()
    #         ax.set_aspect(abs((xmax-xmin)/(ymax-ymin))*ratio[i], adjustable='box-forced')


    # set right and left y axes, all other are invisible
    for i, ax in enumerate(axs):

        make_patch_spines_invisible(ax)
        ax.patch.set_alpha(0)
        if i % 2 == 0:
            ax.tick_params(axis='both', which='both', bottom='off', top='off',
                            left='on', labelleft='on', right='off', labelbottom='off')
            ax.spines['left'].set_visible(True)
            ax.yaxis.tick_left()
        else:
            ax.tick_params(axis='both', which='both', bottom='off', top='off',
                            left='off', right='on',labelright='on', labelbottom='off')
            ax.spines['right'].set_visible(True)
            ax.yaxis.tick_right()

    # add x-axis on top and bottom
    axs[0].spines['top'].set_visible(True)
    axs[0].tick_params(axis='both', which='both', labeltop='on', top='on')
    axs[-1].spines['bottom'].set_visible(True)
    axs[-1].tick_params(axis='both', which='both', labelbottom='on', bottom='on')

    # stacking all axes
    plt.subplots_adjust(hspace=hsapce)

    # change color for all axes
    if color is not None:
        for ax, c in zip(axs, color):
            change_color_spines(ax, c)

    return axs
