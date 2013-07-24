import numpy as np
import scipy.optimize

from ndop.optimization.cost_function import Cost_function

import util.io
from util.debug import print_debug

def optimize(p0, years=7000, tolerance=0, time_step_size=1, maxiter=None, debug_level=0, required_debug_level=1):
    from ndop.metos3d.constants import  MODEL_PARAMETER_LOWER_BOUND, MODEL_PARAMETER_UPPER_BOUND
    from ndop.optimization.constants import P_FILE, F_FILE, F_EVAL_FILE, DF_FILE, DF_EVAL_FILE, RES_FILE
    
    ## construct fun and jac
    cost_function = Cost_function(years=years, tolerance=tolerance, time_step_size=time_step_size, debug_level=debug_level, required_debug_level=required_debug_level + 1)
    
    def save_iteration(x, file, format_string='%.6g'):
        file_npy = file + '.npy'
#         file_txt = file + '.txt'
        
        try:
            iteration = np.load(file_npy)
        except (OSError, IOError):
            iteration = None
        
        x = x.reshape((1,) + x.shape)
        
        if iteration is None:
            iteration = np.copy(x)
        else:
            iteration = np.append(iteration, x, axis=0)
        
        util.io.save_npy_and_txt(iteration, file)
        
#         np.save(file_npy, iteration)
#         np.savetxt(file_txt, iteration, fmt=format_string)
    
    def increment(file):
        file_npy = file + '.npy'
        file_txt = file + '.txt'
        
        try:
            x = np.load(file_npy)
        except (OSError, IOError):
            x = np.array([0])
            
        x += 1
        
        np.save(file_npy, x)
        np.savetxt(file_txt, x, fmt='%i')
    
    
    def fun(p):
        print_debug(('Current p value ', p), debug_level, required_debug_level, 'ndop.optimization.run.optimize: ')
        save_iteration(p, P_FILE)
        
        df = cost_function.df(p)
        f = cost_function.f(p)
        
        print_debug(('Current f value ', f), debug_level, required_debug_level, 'ndop.optimization.run.optimize: ')
        save_iteration(f, F_FILE)
        increment(F_EVAL_FILE)
        
        return f
    
    def jac(p):
        df = cost_function.df(p)
        
        print_debug(('Current df value ', df), debug_level, required_debug_level, 'ndop.optimization.run.optimize: ')
        save_iteration(df, DF_FILE)
        increment(DF_EVAL_FILE)
        
        return df
    
    
    ## construct bounds
    bounds = []
    p_len = len(MODEL_PARAMETER_LOWER_BOUND)
    for i in range(p_len):
        p_i_lower = MODEL_PARAMETER_LOWER_BOUND[i]
        if p_i_lower == - np.inf:
            p_i_lower = None
        
        p_i_upper = MODEL_PARAMETER_UPPER_BOUND[i]
        if p_i_upper == np.inf:
            p_i_upper = None
        
        bounds += [(p_i_lower, p_i_upper)]
    
    
    ## construct options
    options = {}
    if debug_level > 0:
        options['disp'] =  True
    if maxiter is not None:
        options['maxiter'] = maxiter
    
    
    #method = 'SLSQP'
    #method = 'TNC'
    method = 'L-BFGS-B'
    
    res = scipy.optimize.minimize(fun, p0, method=method, jac=jac, bounds=bounds, options=options)
    
    util.io.save_object(res, RES_FILE)
    
    return res