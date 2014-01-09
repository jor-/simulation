import sys
import argparse

import os
import numpy as np

import util.pattern
import util.io


def get_min_cf_value():
    from ndop.metos3d.constants import MODEL_OUTPUTS_DIR, MODEL_TIME_STEP_DIRNAME, MODEL_PARAMETERS_FILENAME
    COST_FUNCTION_OUTPUT_DIRNAME = 'cost_function/2'
    COST_FUNCTION_F_FILENAME = 'cost_function_f.txt'
    
    min_cf_value = float('inf')
    min_cf_parameter_set_number = None
    min_cf_p_str = None
    
    time_step_dirname = util.pattern.replace_int_pattern(MODEL_TIME_STEP_DIRNAME, 1)
    time_step_dir = os.path.join(MODEL_OUTPUTS_DIR, time_step_dirname)
    
    print('Looking for parameter sets in ' + time_step_dir + '.')
    
    parameter_set_dirs = util.io.get_dirs(time_step_dir)
    
    for parameter_set_dir in parameter_set_dirs:
#     for parameter_set_number in parameter_sets:
#         parameter_set_dirname = util.pattern.replace_int_pattern(MODEL_PARAMETERS_SET_DIRNAME, parameter_set_number)
#         parameter_set_dir = os.path.join(time_step_dir, parameter_set_dirname)

        print('Looking at ' + parameter_set_dir + '.')
        
        cost_function_output_path = os.path.join(parameter_set_dir, COST_FUNCTION_OUTPUT_DIRNAME)
        cost_function_f_file = os.path.join(cost_function_output_path, COST_FUNCTION_F_FILENAME)
        
        if os.path.exists(cost_function_f_file):
            print('Looking at ' + cost_function_f_file + '.')
            cf_value = np.sum(np.loadtxt(cost_function_f_file))
            
            if cf_value < min_cf_value:
                min_cf_value = cf_value
                min_cf_parameter_set_dir = parameter_set_dir
                
                parameters_file = os.path.join(parameter_set_dir, MODEL_PARAMETERS_FILENAME)
                min_cf_p = np.loadtxt(parameters_file)
                with open(parameters_file) as f:
                    min_cf_p_str = list(f)
    
    return min_cf_value, min_cf_parameter_set_dir, min_cf_p_str




if __name__ == "__main__":
    cf_value, ps, p_str = get_min_cf_value()
    print('cf_value = ')
    print(cf_value)
    print('ps = ')
    print(ps)
    print('p = ')
    for p_i in p_str:
        print(p_i, end='')