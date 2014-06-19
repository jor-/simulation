import os.path
import numpy as np

import util.io
import util.pattern

from ndop.constants import PARAMETER_OPTIMIZATION_DIR
ITERATIONS_DIRNAME = 'iterations'


def get_values(cf_kind, value_kind):
    ## get files
    dir = os.path.join(PARAMETER_OPTIMIZATION_DIR, cf_kind, ITERATIONS_DIRNAME)
#     pattern = 'iteration_' + value_kind + '_[0-9]{3}.txt'
    pattern = value_kind + '_[0-9]{3}.txt'
    files = util.io.get_files(dir, pattern)
    
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
        value_array = np.ones(shape) * np.nan
        value_array[indices] = values
    else:
        value_array = np.empty(0)
    
    ## return
    return value_array
    