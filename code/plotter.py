import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm



def plot_overdensity_field(tracer_field, normalize=False, vmax=None, 
                      title=None, show_labels=True, show_colorbar=True,
                      zslice_min=None, zslice_max=None, 
                      figsize=(6,6), symlog=False,
                      xlim=None, ylim=None, 
                      tracers=None, alpha=0.5, s=1,
                      box_size=None,
                      ):

        print(np.min(tracer_field), np.max(tracer_field))

        if normalize:
            tracer_field /= np.max(np.abs(tracer_field))
        print(np.min(tracer_field), np.max(tracer_field))
        
        if vmax is None:
            #vmax = np.max(np.abs(tracer_field))
            vmax = 3*np.std(tracer_field)

        if box_size is None:
            print("Box size not passed, using default zslices")
            zslice_min = 0
            zslice_max = 20 #mpc/h
        
        i_zslice_min = int(zslice_min/box_size * tracer_field.shape[-1])
        i_zslice_max = int(zslice_max/box_size * tracer_field.shape[-1])

        field_2d = np.mean(tracer_field[:,:,i_zslice_min:i_zslice_max], axis=-1)
        print(field_2d.shape)

        plt.figure(figsize=figsize, facecolor=(1,1,1,0))
        ax = plt.gca()        
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

        extent = None
        if box_size is not None:
            extent = [0, box_size, 0, box_size]
            ax.set_xlabel(r'$h^{-1}\,\text{Mpc}$')
            ax.set_ylabel(r'$h^{-1}\,\text{Mpc}$')
            
        im = plt.imshow(field_2d, norm=norm, cmap='RdBu', extent=extent)
        
        if tracers is not None:
            ax.scatter(tracers[:,0], tracers[:,1], s=s, color='k', alpha=alpha)
        
        if show_colorbar:
            cbar = plt.colorbar(im, label=r'overdensity $\delta$', fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=12) 
            
        if not show_labels:    
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)    

        plt.show()