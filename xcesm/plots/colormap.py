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


