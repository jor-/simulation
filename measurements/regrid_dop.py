import numpy as np

import ndop.metos3d.data

from util.debug import print_debug

def load_measurements(debug_level = 0, required_debug_level = 1):
    from ndop.measurements.constants import YOSHIMURA_DOP_MEASUREMENT_FILE, LADOLFI_2002_DOP_MEASUREMENT_FILE, LADOLFI_2004_DOP_MEASUREMENT_FILE
    
    ## Yoshimirua data
    print_debug(('Loading dop measurements from ', YOSHIMURA_DOP_MEASUREMENT_FILE), debug_level, required_debug_level,  'ndop.measurements.regrid_dop.measurements_array: ')
    
    measurements_yoshimura = np.loadtxt(YOSHIMURA_DOP_MEASUREMENT_FILE)
    
    ## Ladolfi 2002 CD139 data
    print_debug(('Loading dop measurements from ', LADOLFI_2002_DOP_MEASUREMENT_FILE), debug_level, required_debug_level,  'ndop.measurements.regrid_dop.measurements_array: ')
    
    measurements_ladolfi_1 = np.loadtxt(LADOLFI_2002_DOP_MEASUREMENT_FILE)
    # skip flagged values
    measurements_ladolfi_1 = measurements_ladolfi_1[measurements_ladolfi_1[:, 5] == 1]
    measurements_ladolfi_1 = measurements_ladolfi_1[:, 0:5]
    
    ## Ladolfi 2004 CD139 data
    print_debug(('Loading dop measurements from ', LADOLFI_2004_DOP_MEASUREMENT_FILE), debug_level, required_debug_level,  'ndop.measurements.regrid_dop.measurements_array: ')
    
    # convert missing values to - Inf
    def convert_ladolfi_2_DOP_values(value_string):
        try:
            value = float(value_string)
        except ValueError:
            value = - float('inf')
        
        return value
    
    measurements_ladolfi_2 = np.loadtxt(LADOLFI_2004_DOP_MEASUREMENT_FILE, converters={4: lambda value_string: convert_ladolfi_2_DOP_values(value_string)})
    # skip flagged values
    measurements_ladolfi_2 = measurements_ladolfi_2[measurements_ladolfi_2[:, 5] == 0]
    measurements_ladolfi_2 = measurements_ladolfi_2[:, 0:5]
    # skip negative values
    measurements_ladolfi_2 = measurements_ladolfi_2[measurements_ladolfi_2[:, 4] >= 0]
    
    ## concatenate measurements
    measurements = np.concatenate((measurements_yoshimura, measurements_ladolfi_1, measurements_ladolfi_2))
    
    return measurements




def init_masked_array(land_sea_mask, t_dim, dtype=np.float64):
    z_dim = np.nanmax(land_sea_mask)
    (x_dim, y_dim) = land_sea_mask.shape
    shape = (t_dim, x_dim, y_dim, z_dim,)
    array = np.zeros(shape, dtype=dtype)
    
    for x, y in np.ndindex(x_dim, y_dim):
        z_max = land_sea_mask[x, y]
        array[:, x, y, z_max:] = np.nan

    return array



def save_regrided(land_sea_mask, t_dim=12, debug_level = 0, required_debug_level = 1):
    from ndop.measurements.constants import DOP_NOBS, DOP_VARIS, DOP_MEANS, DOP_MOS
    
    print_debug('Calculating and saving dop measurement data.', debug_level, required_debug_level,  'ndop.measurements.regrid_dop.save_regrided: ')
    
    ## load measurements
    measurements = load_measurements(debug_level, required_debug_level + 1)
    
    ## init values
    nobs = init_masked_array(land_sea_mask, t_dim)
    varis = np.array(nobs, copy=True)
    
    sum_of_values = np.empty(nobs.shape, dtype=np.float64) * np.nan
    sum_of_squares = np.array(sum_of_values, copy=True)
    
    number_of_measurements = measurements.shape[0]
    
    ## insert measurements
    for i in range(number_of_measurements):
        t, x, y, z, dop = measurements[i, :]
        (t_index, x_index, y_index, z_index) = ndop.metos3d.data.get_index(t, x, y, z, t_dim, land_sea_mask, debug_level = debug_level, required_debug_level = required_debug_level + 1)
        
        nobs[t_index, x_index, y_index, z_index] += 1
        
        if np.isnan(sum_of_values[t_index, x_index, y_index, z_index]):
            sum_of_values[t_index, x_index, y_index, z_index] = dop
        else:
            sum_of_values[t_index, x_index, y_index, z_index] += dop
        
        if np.isnan(sum_of_squares[t_index, x_index, y_index, z_index]):
            sum_of_squares[t_index, x_index, y_index, z_index] = dop**2
        else:
            sum_of_squares[t_index, x_index, y_index, z_index] += dop**2
    
    ## average measurements
    where_measurements = np.where(nobs > 0)
    
    mean = np.array(sum_of_values, copy=True)
    mean[where_measurements] /= nobs[where_measurements]
    
    ## calculate variance
    where_measurements_ge_3 = np.where(nobs >= 3)
    nobs_3 = nobs[where_measurements_ge_3]
    sum_of_values_3 = sum_of_values[where_measurements_ge_3]
    sum_of_squares_3 = sum_of_squares[where_measurements_ge_3]
    
    # sd^2 = (SUM(x_i*x_i)-(SUM(x_i))^2/n)/(n-1)
    varis[where_measurements_ge_3] = (sum_of_squares_3 - sum_of_values_3 ** 2 / nobs_3) / (nobs_3 - 1)
    varis[np.where(varis < 0)] = 0
    vari_averaged = np.nansum(varis) / (varis > 0).sum()
    varis[np.where(varis == 0)] = vari_averaged
    
    ## save values
    np.save(DOP_NOBS, nobs)
    np.save(DOP_VARIS, varis)
    np.save(DOP_MEANS, mean)