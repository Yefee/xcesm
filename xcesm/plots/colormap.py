import numpy as np
import matplotlib.colors as mcolors
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

def subplots(nrow=2, ncol=2, figsize=None, ind=None):
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure()

    tol = nrow * ncol   
    if ind is not None:
        pass
    else:
        ind = np.arange(tol) + 1

    projection = ccrs.PlateCarree()
    ax = [fig.add_subplot(nrow, ncol, i, projection=projection) for i in ind]

    return ax

