import numpy as np

import ndop.metos3d.data

from util.debug import print_debug


def measurements_array(debug_level = 0, required_debug_level = 1):
    from ndop.measurements.constants import YOSHIMURA_DOP_MEASUREMENT_FILE
    
    print_debug(('Loading dop measurements from ', YOSHIMURA_DOP_MEASUREMENT_FILE), debug_level, required_debug_level,  'ndop.measurements.regrid_data.measurements_array: ')
    
    x = np.loadtxt(YOSHIMURA_DOP_MEASUREMENT_FILE, usecols = (0,))
    y = np.loadtxt(YOSHIMURA_DOP_MEASUREMENT_FILE, usecols = (1,))
    z = np.zeros_like(x)
    time = np.loadtxt(YOSHIMURA_DOP_MEASUREMENT_FILE, usecols = (2,))
    
    measurements = np.empty([len(time), 4], dtype=np.float64)
    measurements[:, 0] = time
    measurements[:, 1] = x
    measurements[:, 2] = y
    measurements[:, 3] = z
    
    return measurements



def values_array(debug_level = 0, required_debug_level = 1):
    from ndop.measurements.constants import YOSHIMURA_DOP_MEASUREMENT_FILE
    
    print_debug(('Loading dop values from ', YOSHIMURA_DOP_MEASUREMENT_FILE), debug_level, required_debug_level, 'ndop.measurements.regrid_data.values_array: ')
    
    values = np.loadtxt(YOSHIMURA_DOP_MEASUREMENT_FILE, usecols = (3,))
    
    return values



def init_masked_array(land_sea_mask, t_dim, dtype=np.float64):
    z_dim = np.nanmax(land_sea_mask)
    (x_dim, y_dim) = land_sea_mask.shape
    shape = (t_dim, x_dim, y_dim, z_dim,)
    array = np.zeros(shape, dtype=dtype)
    
#     for i in range(z_dim):
#         np.where(land_sea_mask == i)
#         array[:, np.where(land_sea_mask == i), i:] = np.nan
    for x, y in np.ndindex(x_dim, y_dim):
        z_max = land_sea_mask[x, y]
        array[:, x, y, z_max:] = np.nan

    
    return array


def save_regrided(land_sea_mask, t_dim=12, debug_level = 0, required_debug_level = 1):
    from ndop.measurements.constants import DOP_NOBS, DOP_VARIS, DOP_MEANS, DOP_SQUARES
    
    print_debug('Calculating and saving dop measurement data.', debug_level, required_debug_level,  'ndop.measurements.regrid_data.save_regrided: ')
    
    ## load measurements
    measurements = measurements_array(debug_level, required_debug_level + 1)
    values = values_array(debug_level, required_debug_level + 1)
    
    ## init values
    nobs = init_masked_array(land_sea_mask, t_dim)
    varis = np.array(nobs, copy=True)
    means = np.array(nobs, copy=True)
    squares = np.array(nobs, copy=True)
#     z_dim = np.nanmax(land_sea_mask)
#     shape = (t_dim,) + land_sea_mask.shape + (z_dim,)
#     nobs = np.empty(shape, dtype=np.int)
#     for i in range(t_dim):
#         nobs[i, :] = land_sea_mask * 0
#     varis = np.empty(shape, dtype=np.float64)
#     for i in range(t_dim):
#         varis[i, :] = land_sea_mask * 0
#     means = np.array(varis, copy=True) 
#     squares = np.array(varis, copy=True) 
    
    ## insert measurements
    for i in range(len(values)):
        t, x, y, z = measurements[i, :]
        dop = values[i]
        (t_index, x_index, y_index, z_index) = ndop.metos3d.data.get_index(t, x, y, z, t_dim, land_sea_mask, debug_level = 0, required_debug_level = 1)
        
        nobs[t_index, x_index, y_index, z_index] += 1
        means[t_index, x_index, y_index, z_index] += dop
        squares[t_index, x_index, y_index, z_index] += dop**2
    
    ## average measurements
    where_measurements = np.where(nobs >= 1)
    means[where_measurements] /= nobs[where_measurements]
    squares[where_measurements] /= nobs[where_measurements]
    
    ## calculate variance
    where_measurements_ge_3 = np.where(nobs >= 3)
    squares3 = squares[where_measurements_ge_3]
    means3 = means[where_measurements_ge_3]
    nobs3 = nobs[where_measurements_ge_3]
    
    #sd^2 = (SUM(x_i*x_i)-(SUM(x_i))^2/n)/(n-1)
    varis[where_measurements_ge_3] = (squares3 - means3**2) * (nobs3 / (nobs3 - 1))
    varis[np.where(varis < 0)] = 0
    vari_averaged = np.nansum(varis) / (varis > 0).sum()
    varis[np.where(varis == 0)] = vari_averaged
    
    ## save values
    np.save(DOP_NOBS, nobs)
    np.save(DOP_VARIS, varis)
    np.save(DOP_MEANS, means)
    np.save(DOP_SQUARES, squares)



    