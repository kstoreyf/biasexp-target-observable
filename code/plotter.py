import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm



def plot_overdensity_field(tracer_field, normalize=False, vmax=None, 
                      title=None, show_labels=True, show_colorbar=True,
                      slice_width=1, figsize=(6,6), symlog=False):

        print(np.min(tracer_field), np.max(tracer_field))

        if normalize:
            tracer_field /= np.max(np.abs(tracer_field))
        print(np.min(tracer_field), np.max(tracer_field))
        
        if vmax is None:
            #vmax = np.max(np.abs(tracer_field))
            vmax = 3*np.std(tracer_field)

        field_2d = np.mean(tracer_field[0:slice_width,:,:], axis=0)

        plt.figure(figsize=figsize, facecolor=(1,1,1,0))
        plt.title(title, fontsize=16)
        
        if symlog:
            linthresh = 0.1*vmax
            linscale = 1.0
            norm = SymLogNorm(
                    linthresh=linthresh, linscale=linscale,
                    vmin=-vmax, vmax=vmax
                    )
        else:
            norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax)

        im = plt.imshow(field_2d, norm=norm, cmap='RdBu')
        ax = plt.gca()        
        
        if show_colorbar:
            cbar = plt.colorbar(im, label=r'overdensity $\delta$', fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=12) 
            
        if not show_labels:    
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)

        plt.show()