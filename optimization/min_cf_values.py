import sys
import argparse

import os
import numpy as np

import util.pattern
import util.io


COST_FUNCTION_NAMES = ('WOA_OLS', 'WOA_WLS', 'WOD_OLS', 'WOD_WLS', 'WOD_GLS')

def min_cf_values():
    from ndop.model.constants import MODEL_OUTPUT_DIR, MODEL_TIME_STEP_DIRNAME, MODEL_PARAMETERS_FILENAME
    
    for cost_function_name in COST_FUNCTION_NAMES:
        COST_FUNCTION_OUTPUT_DIRNAME = 'cost_functions/' + cost_function_name
        COST_FUNCTION_F_FILENAME = 'f.txt'
        
        min_cf_value = float('inf')
        min_cf_parameter_set_number = None
        min_cf_p_str = None
        
#         time_step_dirname = util.pattern.replace_int_pattern(MODEL_TIME_STEP_DIRNAME, 1)
        time_step_dirname = MODEL_TIME_STEP_DIRNAME.format(1)
        time_step_dir = os.path.join(MODEL_OUTPUT_DIR, time_step_dirname)
        
#         print('Looking for parameter sets in ' + time_step_dir + '.')
        
        parameter_set_dirs = util.io.get_dirs(time_step_dir)
        
        for parameter_set_dir in parameter_set_dirs:
            cost_function_output_path = os.path.join(parameter_set_dir, COST_FUNCTION_OUTPUT_DIRNAME)
            cost_function_f_file = os.path.join(cost_function_output_path, COST_FUNCTION_F_FILENAME)
            
            if os.path.exists(cost_function_f_file):
#                 print('Looking at ' + cost_function_f_file + '.')
                cf_value = np.sum(np.loadtxt(cost_function_f_file))
                
                if cf_value < min_cf_value:
                    min_cf_value = cf_value
                    min_cf_parameter_set_dir = parameter_set_dir
                    
                    parameters_file = os.path.join(parameter_set_dir, MODEL_PARAMETERS_FILENAME)
                    min_cf_p_str = str(np.loadtxt(parameters_file))
#                     with open(parameters_file) as f:
#             else:
#                 print(cost_function_f_file + ' does not exists.')
                
    
        print('For {} has {} with parameters {} the min value {}.'.format(cost_function_name, min_cf_parameter_set_dir, min_cf_p_str, min_cf_value))
#     return min_cf_value, min_cf_parameter_set_dir, min_cf_p_str




if __name__ == "__main__":
    min_cf_values()