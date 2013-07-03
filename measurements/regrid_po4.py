import numpy as np
import bisect

import util.io
from util.debug import print_debug

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
    from ndop.measurements.constants import WOA_PO4_MOS_NETCDF_ANNUAL_FILE, WOA_PO4_MOS_NETCDF_MONTHLY_FILE, WOA_PO4_MOS_NETCDF_DATANAME, PO4_MOS
    
    from ndop.measurements.constants import PO4_ANNUAL_THRESHOLD
    from ndop.metos3d.constants import METOS_Z
    
    
    base_string = 'util.measurements.regrid_po4.save_regrided: '
    
    z_index_annual_threshold = bisect.bisect(METOS_Z, PO4_ANNUAL_THRESHOLD)
    print_debug(('Taking annual data from z index ', z_index_annual_threshold, '.'), debug_level, required_debug_level, base_string)
    
    for (netcdf_annual_file, netcdf_monthly_file, netcdf_dataname, npy_file, divide) in ((WOA_PO4_NOBS_NETCDF_ANNUAL_FILE, WOA_PO4_NOBS_NETCDF_MONTHLY_FILE, WOA_PO4_NOBS_NETCDF_DATANAME, PO4_NOBS, True), (WOA_PO4_VARIS_NETCDF_ANNUAL_FILE, WOA_PO4_VARIS_NETCDF_MONTHLY_FILE, WOA_PO4_VARIS_NETCDF_DATANAME, PO4_VARIS, False), (WOA_PO4_MEANS_NETCDF_ANNUAL_FILE, WOA_PO4_MEANS_NETCDF_MONTHLY_FILE, WOA_PO4_MEANS_NETCDF_DATANAME, PO4_MEANS, False), (WOA_PO4_MOS_NETCDF_ANNUAL_FILE, WOA_PO4_MOS_NETCDF_MONTHLY_FILE, WOA_PO4_MOS_NETCDF_DATANAME, PO4_MOS, False)):
        
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
    