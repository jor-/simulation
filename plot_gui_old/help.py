import measurements.util.map

import numpy as np
import matplotlib.pyplot as plt


def plot_iterate_depth(data, file):
    ## prepare land sea mask
    land_sea_mask_map = measurements.util.map.init_masked_map(default_value=1, dtype=np.float16)
    land_sea_mask_map = land_sea_mask_map.swapaxes(-1, -2)
    land_sea_mask_map = land_sea_mask_map.swapaxes(-2, -3)
    
    ## prepare colormaps
    colormap = plt.cm.jet
    colormap_no_data = plt.cm.gray
    colormap_no_data.set_bad(alpha=0)
    
    ## swap x and y to the last axes
    data = data.swapaxes(-1, -2)
    data = data.swapaxes(-2, -3)
    
    ## prepare figure
    z_dim = 15
    
    fig = plt.figure(figsize=(10, 5*z_dim))
    
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)
    
    for i in range(z_dim):
        ax = fig.add_subplot(z_dim, 1, i+1)
        
        ## plot data
        axes_image = plt.imshow(data[i].transpose(), origin='lower', extent=(0, 360, -90, 90), aspect='equal', cmap=colormap, vmin=vmin, vmax=vmax)
        cb = fig.colorbar(axes_image, ax=ax, fraction=.022, pad=.1, aspect=20)
        
        ## plot no data mask -> land: black (0), missing values: white (1), data values: alpha (nan)
        no_data_map = np.copy(land_sea_mask_map[i])
        no_data_map[np.isnan(land_sea_mask_map[i])] = 0.5
        no_data_map[np.isnan(land_sea_mask_map[0])] = 0
        no_data_map[np.logical_not(np.isnan(data[i]))] = np.nan
        axes_image = plt.imshow(no_data_map.transpose(), origin='lower', extent=(0, 360, -90, 90), aspect='equal', cmap=colormap_no_data, vmin=0, vmax=1)
        
        plt.title('depth layer ' + str(i+1))
    
    plt.savefig(file, bbox_inches='tight')
        
#         plt.xlabel('longitude')
#         plt.ylabel('latitude')
    


# def make_plot(data, file):
#     ## prepare land sea mask
#     land_sea_mask_map = measurements.util.init_masked_map(default_value=1, dtype=np.float16)
#     land_sea_mask_map = land_sea_mask_map.swapaxes(-1, -2)
#     land_sea_mask_map = land_sea_mask_map.swapaxes(-2, -3)
#     
#     ## prepare colormaps
#     colormap = plt.cm.jet
#     colormap_no_data = plt.cm.gray
#     colormap_no_data.set_bad(alpha=0)
#     
#     ## swap x and y to the last axes
#     data = data.swapaxes(-1, -2)
#     data = data.swapaxes(-2, -3)
#     
#     iter_shape = data.shape[:-2]
#     n = 1
#     for i in iter_shape:
#         n *= i
#     
#     vmin = np.nanmin(data)
#     vmax = np.nanmax(data)
#     
#     fig = plt.figure(figsize=(10, 5*n))
#     
#     
#     i = 1
#     for nd_i in np.ndindex(* data.shape[:-3]):
#         for z_i in range(15):
#             plot_subplot(data[nd_i][z_i], fig, (n, 1, i), land_sea_mask_map[z_i], colormap, colormap_no_data, vmin, vmax)
#             i += 1
#     
#     plt.savefig(file, bbox_inches='tight')