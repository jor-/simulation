import argparse
import sys
import os.path
import scipy.io
import numpy as np

import ndop.optimization.cost_function
import util.logging
logger = util.logging.get_logger()

from ndop.optimization.matlab.constants import MATLAB_PARAMETER_FILENAME, MATLAB_F_FILENAME, MATLAB_DF_FILENAME, NODES_MAX_FILENAME, KIND_OF_COST_FUNCTIONS


## parse arguments
parser = argparse.ArgumentParser(description='Evaluating cost function for matlab.')

parser.add_argument('-y', '--years', type=int, help='Number of maximal years for the spinup.')
parser.add_argument('-t', '--tolerance', default=0, type=float, help='The tolerance at which the spinup terminates.')
parser.add_argument('-a', '--and_combination', action='store_true', help='If year and tolerance both have to be satisfied.')
parser.add_argument('-f', '--eval_function_value', action='store_true', help='Save the value of the cost function.')
parser.add_argument('-g', '--eval_grad_value', action='store_true', help='Save the values of the derivative of the cost function.')
parser.add_argument('-p', '--exchange_dir', help='The directory from which to load the parameters and save the values.')
parser.add_argument('-c', '--kind_of_cost_function', choices=KIND_OF_COST_FUNCTIONS, help='The kind of the cost function to chose.')
parser.add_argument('-d', '--debug_logging_file', default='', help='File to store debug informations.')
parser.add_argument('--version', action='version', version='%(prog)s 0.1')

args = vars(parser.parse_args())
eval_function_value = args['eval_function_value']
eval_grad_value = args['eval_grad_value']
exchange_dir = args['exchange_dir']
log_file = args['debug_logging_file']
kind_of_cost_function = args['kind_of_cost_function']
years = args['years']
tolerance = args['tolerance']
if args['and_combination']:
    combination='and'
else:
    combination='or'
spinup_options = {'years':years, 'tolerance':tolerance, 'combination':combination}



with util.logging.Logger(log_file=log_file, disp_stdout=False):
    with np.errstate(invalid='ignore'):
        
        ## calculate file locations
        p_file = os.path.join(exchange_dir, MATLAB_PARAMETER_FILENAME)
        f_file = os.path.join(exchange_dir, MATLAB_F_FILENAME)
        df_file = os.path.join(exchange_dir, MATLAB_DF_FILENAME)
        job_nodes_max_file = os.path.join(exchange_dir, NODES_MAX_FILENAME)
        
        
        ## choose cost function
        df_accuracy_order = 2
        
        data_kind = kind_of_cost_function[:3]
        cf_kind = kind_of_cost_function[4:]
        if kind_of_cost_function[:3] == 'OLD':
            data_kind = kind_of_cost_function[:7]
            cf_kind = kind_of_cost_function[8:]
#             job_setup = {}
#             job_setup['spinup'] = {}
#             job_setup['spinup']['nodes_setup'] = ('westmere', 8, 12)
#             job_setup['derivative'] = {}
#             job_setup['derivative']['nodes_setup'] = ('westmere', 4, 12)
            job_setup=None
        else:
            job_setup=None
        if cf_kind == 'OLS':
            cf_class = ndop.optimization.cost_function.OLS
        elif cf_kind == 'WLS':
            cf_class = ndop.optimization.cost_function.WLS
        elif cf_kind == 'GLS':
            cf_class = ndop.optimization.cost_function.GLS
        elif cf_kind == 'LWLS':
            cf_class = ndop.optimization.cost_function.LWLS
        else:
            raise ValueError('Unknown cf kind {}.'.format(cf_kind))
        cf = cf_class(data_kind, spinup_options, time_step=1, df_accuracy_order=df_accuracy_order, job_setup=job_setup)
        
        
        ## eval cost function
        p = scipy.io.loadmat(p_file, squeeze_me=True)
        p = p['p']
        if eval_grad_value:
            df = {'df': cf.df(p)}
            scipy.io.savemat(df_file, df, oned_as='column')
        
        if eval_function_value:
            f = {'f': cf.f(p)}
            scipy.io.savemat(f_file, f, oned_as='column')