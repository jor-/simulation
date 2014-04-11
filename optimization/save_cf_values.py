import argparse
import os
import numpy as np

import ndop.optimization.cost_function
from ndop.model.eval import Model

import logging
logger = logging.getLogger(__name__)



def save(parameter_sets=range(1000), kind='WOA', max_nodes_file='/work_O2/sunip229/tmp/save_cf_values_max_nodes.txt'):
    from ndop.model.constants import MODEL_OUTPUT_DIR, MODEL_TIME_STEP_DIRNAME, MODEL_PARAMETERS_SET_DIRNAME, MODEL_PARAMETERS_FILENAME, MODEL_SPINUP_DIRNAME
    
    model = Model()
    time_step = 1
    time_step_dirname = MODEL_TIME_STEP_DIRNAME.format(time_step)
    time_step_dir = os.path.join(MODEL_OUTPUT_DIR, time_step_dirname)
    
    for parameter_set_number in parameter_sets:
        parameter_set_dirname = MODEL_PARAMETERS_SET_DIRNAME.format(parameter_set_number)
        parameter_set_dir = os.path.join(time_step_dir, parameter_set_dirname)
        parameters_file = os.path.join(parameter_set_dir, MODEL_PARAMETERS_FILENAME)
        
        if os.path.exists(parameters_file):
            p = np.loadtxt(parameters_file)
            
            spinup_dir = os.path.join(parameter_set_dir, MODEL_SPINUP_DIRNAME)
            last_run_dir = model.get_last_run_dir(spinup_dir)
            
            years = model.get_total_years(last_run_dir)
            tolerance = model.get_real_tolerance(last_run_dir)
            time_step = model.get_time_step(last_run_dir)
            
            model_f_file = os.path.join(parameter_set_dir, 'F_0012.npy')
            model_df_1_file = os.path.join(parameter_set_dir, 'DF_0012_1.npy')
            model_df_2_file = os.path.join(parameter_set_dir, 'DF_0012_2.npy')
            
            if os.path.isfile(model_df_2_file):
                df_accuracy_order = 2
            else:
                df_accuracy_order = 1
            
            ## create cost functions
            if kind == 'WOA':
                cost_function_family = ndop.optimization.cost_function.WOA_Family(ndop.optimization.cost_function.WOA_WLS, years=years, tolerance=tolerance, combination='and', df_accuracy_order=df_accuracy_order, max_nodes_file=max_nodes_file)
            elif kind == 'WOD':
                cost_function_family = ndop.optimization.cost_function.WOD_Family(ndop.optimization.cost_function.WOD_GLS, years=years, tolerance=tolerance, combination='and', df_accuracy_order=df_accuracy_order, max_nodes_file=max_nodes_file)
            
            cost_function_family.get_f(p)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculating cost function values.')
    
    parser.add_argument('-k', '--kind_of_cost_function', choices=('WOA', 'WOD'), help='The kind of the cost function to chose.')
    parser.add_argument('-f', '--first', type=int, default=0, help='First parameter set number for which to calculate the values.')
    parser.add_argument('-l', '--last', type=int, default=1000, help='Last parameter set number for which to calculate the values.')
    parser.add_argument('--version', action='version', version='%(prog)s 0.1')
    
    args = vars(parser.parse_args())
    kind = args['kind_of_cost_function']
    first = args['first']
    last = args['last']
    
    parameter_sets = range(first, last+1)
    
    logging.basicConfig(level=logging.DEBUG)
    save(parameter_sets=parameter_sets, kind=kind)