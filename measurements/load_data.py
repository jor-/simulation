import numpy as np

import util.io
from util.debug import print_debug

import ndop.measurements.regrid_dop
import ndop.metos3d.data

def po4_npy_or_netcdf(npy_file, netcdf_file, netcdf_dataname, debug_level = 0, required_debug_level = 1):
    base_string = 'util.measurements.load_data.npy_or_netcdf: '
    try:
        print_debug(('Loading data from ', npy_file), debug_level, required_debug_level, base_string)
        
        data = np.load(npy_file)
        
    except IOError:
        print_debug(('File ', npy_file, ' does not exists. Loading data from ', netcdf_file, '.'), debug_level, required_debug_level, base_string)
        
        data = util.io.load_netcdf(netcdf_file, netcdf_dataname, debug_level, required_debug_level + 1)
        data = np.swapaxes(data, 1, 3)  # netcdf shape: (12, 15, 64, 128)
        
        print_debug(('Saving data to ', npy_file), debug_level, required_debug_level, base_string)
        
        np.save(npy_file, data)
    
    return data


def dop_npy_or_save(npy_file, debug_level = 0, required_debug_level = 1):
    base_string = 'util.measurements.load_data.dop_npy_or_save: '
    try:
        print_debug(('Loading data from ', npy_file), debug_level, required_debug_level, base_string)
        
        data = np.load(npy_file)
        
    except IOError:
        print_debug(('File ', npy_file, ' does not exists. Calculating data.'), debug_level, required_debug_level, base_string)
        
        land_sea_mask = ndop.metos3d.data.load_land_sea_mask(debug_level, required_debug_level + 1)
        
        ndop.measurements.regrid_dop.save_regrided(land_sea_mask, t_dim=12, debug_level=debug_level, required_debug_level=required_debug_level+1)
        
        data = np.load(npy_file)
    
    return data



def po4_nobs(debug_level = 0, required_debug_level = 1):
    from ndop.measurements.constants import PO4_NOBS, WOA_PO4_NOBS_NETCDF_FILE, WOA_PO4_NOBS_NETCDF_DATANAME
    
    data = po4_npy_or_netcdf(PO4_NOBS, WOA_PO4_NOBS_NETCDF_FILE, WOA_PO4_NOBS_NETCDF_DATANAME, debug_level, required_debug_level)
    
    return data



def po4_varis(debug_level = 0, required_debug_level = 1):
    from ndop.measurements.constants import PO4_VARIS, WOA_PO4_VARIS_NETCDF_FILE, WOA_PO4_VARIS_NETCDF_DATANAME
    
    data = po4_npy_or_netcdf(PO4_VARIS, WOA_PO4_VARIS_NETCDF_FILE, WOA_PO4_VARIS_NETCDF_DATANAME, debug_level, required_debug_level)
    
    return data



def po4_means(debug_level = 0, required_debug_level = 1):
    from ndop.measurements.constants import PO4_MEANS, WOA_PO4_MEANS_NETCDF_FILE, WOA_PO4_MEANS_NETCDF_DATANAME
    
    data = po4_npy_or_netcdf(PO4_MEANS, WOA_PO4_MEANS_NETCDF_FILE, WOA_PO4_MEANS_NETCDF_DATANAME, debug_level, required_debug_level)
    
    return data


def po4_squares(debug_level = 0, required_debug_level = 1):
    from ndop.measurements.constants import PO4_SQUARES, WOA_PO4_SQUARES_NETCDF_FILE, WOA_PO4_SQUARES_NETCDF_DATANAME
    
    data = po4_npy_or_netcdf(PO4_SQUARES, WOA_PO4_SQUARES_NETCDF_FILE, WOA_PO4_SQUARES_NETCDF_DATANAME, debug_level, required_debug_level)
    
    return data



def dop_nobs(debug_level = 0, required_debug_level = 1):
    from ndop.measurements.constants import DOP_NOBS
    
    data = dop_npy_or_save(DOP_NOBS, debug_level, required_debug_level)
    
    return data


def dop_varis(debug_level = 0, required_debug_level = 1):
    from ndop.measurements.constants import DOP_VARIS
    
    data = dop_npy_or_save(DOP_VARIS, debug_level, required_debug_level)
    
    return data


def dop_means(debug_level = 0, required_debug_level = 1):
    from ndop.measurements.constants import DOP_MEANS
    
    data = dop_npy_or_save(DOP_MEANS, debug_level, required_debug_level)
    
    return data


def dop_squares(debug_level = 0, required_debug_level = 1):
    from ndop.measurements.constants import DOP_SQUARES
    
    data = dop_npy_or_save(DOP_SQUARES, debug_level, required_debug_level)
    
    return data



def append_dop_and_po4(dop, po4, debug_level = 0, required_debug_level = 1):
    dop = dop.reshape((1,) + dop.shape)
    po4 = po4.reshape((1,) + po4.shape)
    
    both = np.append(dop, po4, axis=0)
    
    return both


def nobs(debug_level = 0, required_debug_level = 1):
    dop = dop_nobs(debug_level, required_debug_level)
    po4 = po4_nobs(debug_level, required_debug_level)
    both = append_dop_and_po4(dop, po4, debug_level, required_debug_level)
    return both


def means(debug_level = 0, required_debug_level = 1):
    dop = dop_means(debug_level, required_debug_level)
    po4 = po4_means(debug_level, required_debug_level)
    both = append_dop_and_po4(dop, po4, debug_level, required_debug_level)
    return both


def squares(debug_level = 0, required_debug_level = 1):
    dop = dop_squares(debug_level, required_debug_level)
    po4 = po4_squares(debug_level, required_debug_level)
    both = append_dop_and_po4(dop, po4, debug_level, required_debug_level)
    return both


def varis(debug_level = 0, required_debug_level = 1):
    dop = dop_varis(debug_level, required_debug_level)
    po4 = po4_varis(debug_level, required_debug_level)
    both = append_dop_and_po4(dop, po4, debug_level, required_debug_level)
    return both