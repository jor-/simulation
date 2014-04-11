import argparse
import os.path
import scipy.io

import ndop.optimization.cost_function
import util.logging

from ndop.optimization.matlab.constants import MATLAB_PARAMETER_FILENAME, MATLAB_F_FILENAME, MATLAB_DF_FILENAME, MAX_NODES_FILENAME

KIND_OF_COST_FUNCTIONS=('WOA_LS', 'WOA_WLS', 'WOD_LS', 'WOD_WLS', 'WOD_GLS')

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
years = args['years']
tolerance = args['tolerance']
eval_function_value = args['eval_function_value']
eval_grad_value = args['eval_grad_value']
exchange_dir = args['exchange_dir']
logging_file = args['debug_logging_file']
kind_of_cost_function = args['kind_of_cost_function']
if args['and_combination']:
    combination='and'
else:
    combination='or'
    

# if len(debug_logging_file) > 0:
#     logging.basicConfig(filename=debug_logging_file,level=logging.DEBUG)


with util.logging.Logger(logging_file=logging_file):
    
    ## calculate file locations
    p_file = os.path.join(exchange_dir, MATLAB_PARAMETER_FILENAME)
    f_file = os.path.join(exchange_dir, MATLAB_F_FILENAME)
    df_file = os.path.join(exchange_dir, MATLAB_DF_FILENAME)
    max_nodes_file = os.path.join(exchange_dir, MAX_NODES_FILENAME)
    
    
    ## choose cost function
    df_accuracy_order = 2
    if kind_of_cost_function == 'WOA_LS':
        cf = ndop.optimization.cost_function.WOA_Family(ndop.optimization.cost_function.WOA_LS, years=years, tolerance=tolerance, combination=combination, max_nodes_file=max_nodes_file, df_accuracy_order=df_accuracy_order)
    elif kind_of_cost_function == 'WOA_WLS':
        cf = ndop.optimization.cost_function.WOA_Family(ndop.optimization.cost_function.WOA_WLS, years=years, tolerance=tolerance, combination=combination, max_nodes_file=max_nodes_file, df_accuracy_order=df_accuracy_order)
    elif kind_of_cost_function == 'WOD_LS':
        cf = ndop.optimization.cost_function.WOD_Family(ndop.optimization.cost_function.WOD_LS, years=years, tolerance=tolerance, combination=combination, max_nodes_file=max_nodes_file, df_accuracy_order=df_accuracy_order)
    elif kind_of_cost_function == 'WOD_WLS':
        cf = ndop.optimization.cost_function.WOD_Family(ndop.optimization.cost_function.WOD_WLS, years=years, tolerance=tolerance, combination=combination, max_nodes_file=max_nodes_file, df_accuracy_order=df_accuracy_order)
    elif kind_of_cost_function == 'WOD_GLS':
        cf = ndop.optimization.cost_function.WOD_Family(ndop.optimization.cost_function.WOD_GLS, years=years, tolerance=tolerance, combination=combination, max_nodes_file=max_nodes_file, df_accuracy_order=df_accuracy_order)
#     cf = Cost_Function(years=years, tolerance=tolerance, max_nodes_file=max_nodes_file)
    
    
    ## eval cost function
    p = scipy.io.loadmat(p_file, squeeze_me=True)
    p = p['p']
    
    if eval_grad_value:
        df = {}
        df['df'] = cf.get_df(p)
        scipy.io.savemat(df_file, df, oned_as='column')
    
    if eval_function_value:
        f = {}
        f['f'] = cf.get_f(p)
        scipy.io.savemat(f_file, f, oned_as='column')