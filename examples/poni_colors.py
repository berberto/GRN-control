import numpy as np
from matplotlib.colors import to_rgb, ListedColormap
from matplotlib import use
use("Agg")
import matplotlib.colors as mcol
poni_colors_dict = {
    "Shh":   "#000000", # black
    "Ptch":  "#8D8D8D", # grey
    "GliFL": "#8900FF", # purple
    "GliA":  "#00AD2F", # green
    "GliR":  "#D02B09", # red
    "Pax":   "#0040FF", # blue
    "Olig":  "#FF00EB", # magenta
    "Nkx":   "#02DDD3", # cyan
    "Irx":   "#DDB002"  # gold
}
poni_colors_array = np.array([mcol.to_rgb(poni_colors_dict[gene]) 
                                for gene in ['Pax', 'Olig', 'Nkx', 'Irx', 'Shh', 'GliA', 'GliR']
                             ])

green = poni_colors_dict["GliA"]
red = poni_colors_dict["GliR"]

def cmap(color):
    rgb = to_rgb(color)
    N = 256
    vals = np.zeros((N,4))
    vals[:,-1] = np.ones(N)
    for i in range(3):
        # from white to color
        vals[:,i] = np.linspace(rgb[i],1,N)[::-1]

        # from black to color
        # vals[:,i] = np.linspace(0,rgb[i],N)
    cmap = ListedColormap(vals)
    return cmap



