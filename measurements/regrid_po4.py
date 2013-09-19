import numpy as np
import bisect

import util.io
from util.debug import print_debug

import scipy.interpolate



def load_from_netcdf(netcdf_file, netcdf_dataname, debug_level = 0, required_debug_level = 1):
    base_string = 'util.measurements.regrid_po4.load_from_netcdf: '
    print_debug(('Loading data from ', netcdf_file), debug_level, required_debug_level, base_string)
    
    data = util.io.load_netcdf(netcdf_file, netcdf_dataname, debug_level, required_debug_level + 1)
    data = np.swapaxes(data, 1, 3)  # netcdf shape: (12, 15, 64, 128)
    
    return data

def save_regrided(debug_level = 0, required_debug_level = 1):
    from ndop.measurements.constants import WOA_PO4_NOBS_NETCDF_ANNUAL_FILE, WOA_PO4_NOBS_NETCDF_MONTHLY_FILE, WOA_PO4_NOBS_NETCDF_DATANAME, PO4_NOBS
    from ndop.measurements.constants import WOA_PO4_VARIS_NETCDF_ANNUAL_FILE, WOA_PO4_VARIS_NETCDF_MONTHLY_FILE, WOA_PO4_VARIS_NETCDF_DATANAME, PO4_VARIS
    from ndop.measurements.constants import WOA_PO4_MEANS_NETCDF_ANNUAL_FILE, WOA_PO4_MEANS_NETCDF_MONTHLY_FILE, WOA_PO4_MEANS_NETCDF_DATANAME, PO4_MEANS
#     from ndop.measurements.constants import WOA_PO4_MOS_NETCDF_ANNUAL_FILE, WOA_PO4_MOS_NETCDF_MONTHLY_FILE, WOA_PO4_MOS_NETCDF_DATANAME, PO4_MOS
    
    from ndop.measurements.constants import PO4_ANNUAL_THRESHOLD
    from ndop.metos3d.constants import METOS_Z
    
    
    base_string = 'util.measurements.regrid_po4.save_regrided: '
    
    ## concatenate annual and montly WOA data
    z_index_annual_threshold = bisect.bisect(METOS_Z, PO4_ANNUAL_THRESHOLD)
    print_debug(('Taking annual data from z index ', z_index_annual_threshold, '.'), debug_level, required_debug_level, base_string)
    
#     for (netcdf_annual_file, netcdf_monthly_file, netcdf_dataname, npy_file, divide) in ((WOA_PO4_NOBS_NETCDF_ANNUAL_FILE, WOA_PO4_NOBS_NETCDF_MONTHLY_FILE, WOA_PO4_NOBS_NETCDF_DATANAME, PO4_NOBS, True), (WOA_PO4_VARIS_NETCDF_ANNUAL_FILE, WOA_PO4_VARIS_NETCDF_MONTHLY_FILE, WOA_PO4_VARIS_NETCDF_DATANAME, PO4_VARIS, False), (WOA_PO4_MEANS_NETCDF_ANNUAL_FILE, WOA_PO4_MEANS_NETCDF_MONTHLY_FILE, WOA_PO4_MEANS_NETCDF_DATANAME, PO4_MEANS, False), (WOA_PO4_MOS_NETCDF_ANNUAL_FILE, WOA_PO4_MOS_NETCDF_MONTHLY_FILE, WOA_PO4_MOS_NETCDF_DATANAME, PO4_MOS, False)):
    for (netcdf_annual_file, netcdf_monthly_file, netcdf_dataname, npy_file, divide) in ((WOA_PO4_NOBS_NETCDF_ANNUAL_FILE, WOA_PO4_NOBS_NETCDF_MONTHLY_FILE, WOA_PO4_NOBS_NETCDF_DATANAME, PO4_NOBS, True), (WOA_PO4_VARIS_NETCDF_ANNUAL_FILE, WOA_PO4_VARIS_NETCDF_MONTHLY_FILE, WOA_PO4_VARIS_NETCDF_DATANAME, PO4_VARIS, False), (WOA_PO4_MEANS_NETCDF_ANNUAL_FILE, WOA_PO4_MEANS_NETCDF_MONTHLY_FILE, WOA_PO4_MEANS_NETCDF_DATANAME, PO4_MEANS, False)):
        
        print_debug(('Preparing ', npy_file, '.'), debug_level, required_debug_level, base_string)
        
        data_monthly = load_from_netcdf(netcdf_monthly_file, netcdf_dataname, debug_level, required_debug_level + 1)
        data_annual = load_from_netcdf(netcdf_annual_file, netcdf_dataname, debug_level, required_debug_level + 1)
        
        t_dim = data_monthly.shape[0]
        if divide:
            factor = 1 / t_dim
        else:
            factor = 1
        
        for t in range(t_dim):
            data_monthly[t,:,:,z_index_annual_threshold:] = data_annual[0,:,:,z_index_annual_threshold:] * factor
        
        np.save(npy_file, data_monthly)
    
    
    ## interpolate variance ##
    
    ## prepare available data
    interpolate_threshold = 3
    nobs = np.load(PO4_NOBS)
    vari = np.load(PO4_VARIS)
    
    data_index_t, data_index_x, data_index_y, data_index_z = np.where(nobs >= interpolate_threshold)
    data_points = (data_index_t, data_index_x, data_index_y, data_index_z)
    data_values = vari[data_points]
    
    ## interpolate linear
    print_debug(('Linear interpolating variance.'), debug_level, required_debug_level, base_string)
    
    interpolate_index_t, interpolate_index_x, interpolate_index_y, interpolate_index_z = np.where(np.logical_and(nobs >= 0, nobs < interpolate_threshold))
    interpolate_points = (interpolate_index_t, interpolate_index_x, interpolate_index_y, interpolate_index_z)
    
    interpolate_values = scipy.interpolate.griddata(data_points, data_values, interpolate_points, method='linear')
    interpolate_values[np.logical_or(interpolate_values <= 0, np.isnan(interpolate_values))] = 0
    
    vari[interpolate_points] = interpolate_values
    
    ## interpolate nearest
    print_debug(('Interpolating variance with nearest value.'), debug_level, required_debug_level, base_string)
    
    data_index_t, data_index_x, data_index_y, data_index_z = np.where(vari > 0)
    
    interpolate_index_t, interpolate_index_x, interpolate_index_y, interpolate_index_z = np.where(vari == 0)
    interpolate_points = (interpolate_index_t, interpolate_index_x, interpolate_index_y, interpolate_index_z)
    
    interpolate_values = scipy.interpolate.griddata(data_points, data_values, interpolate_points, method='nearest')
    
    vari[interpolate_points] = interpolate_values
    
    ## saving interpolated variance
    print_debug(('Saving interpolated variance.'), debug_level, required_debug_level, base_string)
    np.save(PO4_VARIS, vari)
    
    
    
    
    
    
#     
#     nobs = np.load(PO4_NOBS)
#     vari = np.load(PO4_VARIS)
#     
#     data_index_t, data_index_x, data_index_y, data_index_z = np.where(nobs >= 3)
#     data_points = (data_index_t, data_index_x, data_index_y, data_index_z)
#     data_values = vari[nobs >= 3]
#     
#     all_index_t, all_index_x, all_index_y, all_index_z = np.where(nobs >= 0)
#     all_points = (all_index_t, all_index_x, all_index_y, all_index_z)
#     
#     print_debug(('Linear interpolating variance.'), debug_level, required_debug_level, base_string)
#     data_values = scipy.interpolate.griddata(data_points, data_values, all_points, method='linear')
#     print_debug(('Interpolating variance with nearest value.'), debug_level, required_debug_level, base_string)
#     data_values = scipy.interpolate.griddata(data_points, data_values, all_points, method='nearest')
#     
#     print_debug(('Saving interpolated variance.'), debug_level, required_debug_level, base_string)
#     np.save(PO4_VARIS, data_values)
#     
#     
#     ## interpolate variance (not linear sperical!)
#     nobs = np.load(PO4_NOBS)
#     vari = np.load(PO4_VARIS)
#     z_array = np.array(METOS_Z)
#     
#     data_index_t, data_index_x, data_index_y, data_index_z = np.where(nobs >= 3)
#     data_z_value = z_array[data_index_z]
#     data_points = (data_index_t, data_index_x, data_index_y, data_z_value)
#     data_values = vari[nobs >= 3]
#     
#     all_index_t, all_index_x, all_index_y, all_index_z = np.where(nobs >= 0)
#     all_z_value = z_array[all_index_z]
#     all_points = (all_index_t, all_index_x, all_index_y, all_z_value)
#     
#     print_debug(('Linear interpolating variance.'), debug_level, required_debug_level, base_string)
#     data_values = scipy.interpolate.griddata(data_points, data_values, all_points, method='linear')
#     print_debug(('Interpolating variance with nearest value.'), debug_level, required_debug_level, base_string)
#     data_values = scipy.interpolate.griddata(data_points, data_values, all_points, method='nearest')
#     
#     print_debug(('Saving interpolated variance.'), debug_level, required_debug_level, base_string)
#     np.save(PO4_VARIS, data_values)
    