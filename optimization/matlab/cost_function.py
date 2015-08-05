import argparse
import sys
import os.path
import tempfile

import numpy as np

import ndop.optimization.cost_function
import ndop.optimization.job
import util.io.matlab
import util.io.fs

import util.logging
logger = util.logging.logger

from ndop.optimization.matlab.constants import MATLAB_PARAMETER_FILENAME, MATLAB_F_FILENAME, MATLAB_DF_FILENAME, NODES_MAX_FILENAME, KIND_OF_COST_FUNCTIONS


## parse arguments
parser = argparse.ArgumentParser(description='Evaluating cost function for matlab.')

parser.add_argument('-c', '--kind_of_cost_function', choices=KIND_OF_COST_FUNCTIONS, help='The kind of the cost function to chose.')

parser.add_argument('-y', '--years', type=int, help='Number of maximal years for the spinup.')
parser.add_argument('-t', '--tolerance', default=0, type=float, help='The tolerance at which the spinup terminates.')
parser.add_argument('-a', '--and_combination', action='store_true', help='If year and tolerance both have to be satisfied.')

parser.add_argument('-f', '--eval_function_value', action='store_true', help='Save the value of the cost function.')
parser.add_argument('-g', '--eval_grad_value', action='store_true', help='Save the values of the derivative of the cost function.')

parser.add_argument('-p', '--exchange_dir', help='The directory from which to load the parameters and save the values.')
parser.add_argument('-d', '--debug_logging_file', default=None, help='File to store debug informations.')

# parser.add_argument('--correlation_min_values', type=int, default=5, help='Number of min values for correlation.')
# parser.add_argument('--correlation_max_year_diff', type=int, default=1, help='Number of max years for correlation.')

parser.add_argument('--version', action='version', version='%(prog)s 0.1')


args = parser.parse_args()

kind_of_cost_function = args.kind_of_cost_function

years = args.years
tolerance = args.tolerance
if args.and_combination:
    combination='and'
else:
    combination='or'

eval_function_value = args.eval_function_value
eval_grad_value = args.eval_grad_value

exchange_dir = args.exchange_dir
log_file = args.debug_logging_file

# correlation_min_values = args.correlation_min_values
# correlation_max_year_diff = args.correlation_max_year_diff

spinup_options = {'years':years, 'tolerance':tolerance, 'combination':combination}
time_step = 1
df_accuracy_order = 2    


with util.logging.Logger(log_file=log_file, disp_stdout=log_file is None):
    with np.errstate(invalid='ignore'):
        
        ## calculate file locations
        p_file = os.path.join(exchange_dir, MATLAB_PARAMETER_FILENAME)
        f_file = os.path.join(exchange_dir, MATLAB_F_FILENAME)
        df_file = os.path.join(exchange_dir, MATLAB_DF_FILENAME)
        # job_nodes_max_file = os.path.join(exchange_dir, NODES_MAX_FILENAME)
        
        
        ## choose cost function
        kind_of_cost_function_splitted = kind_of_cost_function.split('.')
        data_kind = kind_of_cost_function_splitted[0]
        cf_kind = kind_of_cost_function_splitted[1]
        
        # data_kind = kind_of_cost_function[:3]
        # cf_kind = kind_of_cost_function[4:]
        # job_setup = None
        # if kind_of_cost_function[:3] == 'OLD':
        #     data_kind = kind_of_cost_function[:7]
        #     cf_kind = kind_of_cost_function[8:]
        if cf_kind == 'OLS':
            cf_class = ndop.optimization.cost_function.OLS
        elif cf_kind == 'WLS':
            cf_class = ndop.optimization.cost_function.WLS
        elif cf_kind == 'LWLS':
            cf_class = ndop.optimization.cost_function.LWLS
        elif cf_kind == 'GLS':
            cf_class = ndop.optimization.cost_function.GLS
            correlation_min_values = int(kind_of_cost_function_splitted[2])
            correlation_max_year_diff = int(kind_of_cost_function_splitted[3])
            if correlation_max_year_diff < 0:
                correlation_max_year_diff = float('inf')
        # elif cf_kind == 'OLD_LWLS':
        #     cf_class = ndop.optimization.cost_function.OLD_LWLS
        else:
            raise ValueError('Unknown cf kind {}.'.format(cf_kind))
    
        cf_kargs = {'data_kind': data_kind, 'spinup_options':spinup_options, 'time_step':time_step, 'df_accuracy_order':df_accuracy_order}
        if cf_kind == 'GLS':
            cf_kargs['correlation_min_values'] = correlation_min_values
            cf_kargs['correlation_max_year_diff'] = correlation_max_year_diff
        
        # cf = cf_class(data_kind, spinup_options, time_step=time_step, df_accuracy_order=df_accuracy_order, job_setup=job_setup)        
        cf = cf_class(**cf_kargs)
        
        ## load parameter
        parameters = util.io.matlab.load(p_file)
        parameters = parameters['p']
        
        
        ## if necessary start calculation job
        if (eval_function_value and not cf.f_available(parameters)) or (eval_grad_value and not cf.df_available(parameters)):
            from ndop.model.constants import MODEL_START_FROM_CLOSEST_PARAMETER_SET
            from util.constants import TMP_DIR
            
            ## start spinup job
            parameter_set_dir = cf.data_base.model.get_parameter_set_dir(time_step, parameters, create=True)
            spinup_run_dir = cf.data_base.model.get_spinup_run_dir(parameter_set_dir, spinup_options, start_from_closest_parameters=MODEL_START_FROM_CLOSEST_PARAMETER_SET)
            
            ## start cf calculation job
            # with tempfile.TemporaryDirectory(dir=MODEL_TMP_DIR, prefix='cost_function_tmp_') as output_dir:
            #     with ndop.optimization.job.CostFunctionJob(output_dir, parameters, cf_kind, eval_f=eval_function_value, eval_df=eval_grad_value, write_output_file=True, **cf_kargs) as cf_job:
            #         cf_job.start()
            #         cf_job.wait_until_finished()
            output_dir = tempfile.mkdtemp(dir=TMP_DIR, prefix='cost_function_tmp_')
            with ndop.optimization.job.CostFunctionJob(output_dir, parameters, cf_kind, eval_f=eval_function_value, eval_df=eval_grad_value, write_output_file=True, **cf_kargs) as cf_job:
                cf_job.start()
                cf_job.wait_until_finished()
            util.io.fs.remove_recursively(output_dir)
        
        
        ## load cost function values
        if eval_grad_value:
            util.io.matlab.save(df_file, cf.df(parameters), value_name='df', oned_as='column')
        
        if eval_function_value:
            util.io.matlab.save(f_file, cf.f(parameters), value_name='f', oned_as='column')
        