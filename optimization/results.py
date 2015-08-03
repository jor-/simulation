import os.path
import numpy as np

import util.io.fs
import util.pattern

from ndop.constants import PARAMETER_OPTIMIZATION_DIR
ITERATIONS_DIRNAME = 'iterations'
SETUP_DIRNAME = 'setup'


def p_bounds(cf_kind):
    pb_file = os.path.join(PARAMETER_OPTIMIZATION_DIR, cf_kind, SETUP_DIRNAME, 'pb.txt')
    return np.loadtxt(pb_file)



def get_values(cf_kind, value_kind, dtype=np.float64):
    ## get files
    dir = os.path.join(PARAMETER_OPTIMIZATION_DIR, cf_kind, ITERATIONS_DIRNAME)
#     pattern = 'iteration_' + value_kind + '_[0-9]{3}.txt'
    pattern = value_kind + '_[0-9]{3}.txt'
    files = util.io.fs.get_files(dir, pattern)
    
    ## load indices and values
    indices = []
    values = []
    
    for file in files:
        filename = os.path.split(file)[1]
        index = util.pattern.get_int_in_string(filename)
        indices.append(index)
        
        value = np.loadtxt(file)
        values.append(value)
    
    ## make array
    if len(indices) > 0:
        n = max(indices) + 1
        shape = (n,) + values[0].shape
#         value_array = np.empty(shape, dtype) * np.nan
        value_array = np.ma.masked_all(shape, dtype)
        value_array[indices] = values
    else:
        value_array = np.ma.masked_all((0,0), dtype)
#         value_array = np.empty(0, dtype)
    
    value_array = np.ma.masked_invalid(value_array)
    ## return
    return value_array



def all_p(cf_kind):
    return get_values(cf_kind, 'all_p', dtype=np.float64)

def all_f(cf_kind):
    return get_values(cf_kind, 'all_f', dtype=np.float64)

def all_df(cf_kind):
    return get_values(cf_kind, 'all_df', dtype=np.float64)

def solver_p(cf_kind):
    return get_values(cf_kind, 'solver_p', dtype=np.float64)

def solver_f(cf_kind):
    return get_values(cf_kind, 'solver_f', dtype=np.float64)

def solver_df(cf_kind):
    return all_df(cf_kind)[solver_f_indices(cf_kind)]

def solver_f_indices(cf_kind):
    return get_values(cf_kind, 'solver_eval_f_index', dtype=np.int32)

# def local_solver_stop_mask(cf_kind):
#     i = solver_f_indices(cf_kind)
#     df = all_df(cf_kind)
# #     return np.any(np.isnan(df[i]), axis=1)
#     return np.any(df.mask[i], axis=1)
# 
# def local_solver_stop_solver_incides(cf_kind):
#     return np.where(local_solver_stop_mask(cf_kind))[0]

def local_solver_stop_slices(cf_kind):
    df = all_df(cf_kind)
    i = solver_f_indices(cf_kind)
    i = i[i < len(df)]
    stop_indices = np.where(np.any(df.mask[i], axis=1))[0]
    
    start_indices = [0,] + (stop_indices + 1).tolist()
    stop_indices = (stop_indices + 1).tolist() + [None,]
    
    slices = []
    for i in range(len(stop_indices)):
        slices.append(slice(start_indices[i], stop_indices[i]))
    return slices
    
    


