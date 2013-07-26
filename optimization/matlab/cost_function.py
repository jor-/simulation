import argparse
import os.path

import scipy.io

from ndop.optimization.cost_function import Cost_function

from ndop.optimization.matlab.constants import MATLAB_PARAMETER_FILENAME, MATLAB_F_FILENAME, MATLAB_DF_FILENAME


## parse arguments
parser = argparse.ArgumentParser(description='Evaluating cost function for matlab.')

parser.add_argument('-y', '--years', default=3500, type=int, help='Number of maximal years for the spinup.')
parser.add_argument('-t', '--tolerance', default=0.001, type=float, help='The tolerance at which the spinup terminates.')
parser.add_argument('-s', '--time_step_size', default=48, type=int, help='The time step size for the tracer transport.')

parser.add_argument('-f', '--eval_function_value', action='store_true', help='Save the value of the cost function.')
parser.add_argument('-g', '--eval_grad_value', action='store_true', help='Save the values of the derivative of the cost function.')

parser.add_argument('-p', '--exchange_dir', help='The directory from which to load the parameters and save the values.')

parser.add_argument('-d', '--debug_level', default=0, type=int, help='Increase the debug level for more debug informations.')
parser.add_argument('--version', action='version', version='%(prog)s 0.1')

args = vars(parser.parse_args())
years = args['years']
tolerance = args['tolerance']
time_step_size = args['time_step_size']
eval_function_value = args['eval_function_value']
eval_grad_value = args['eval_grad_value']
exchange_dir = args['exchange_dir']
debug_level = args['debug_level']


## calculate file locations
p_file = os.path.join(exchange_dir, MATLAB_PARAMETER_FILENAME)
f_file = os.path.join(exchange_dir, MATLAB_F_FILENAME)
df_file = os.path.join(exchange_dir, MATLAB_DF_FILENAME)


## eval cost function
cf = Cost_function(years=years, tolerance=tolerance, time_step_size=time_step_size, debug_level=debug_level)

p = scipy.io.loadmat(p_file, squeeze_me=True)
p = p['p']

if eval_grad_value:
    df = {}
    df['df'] = cf.df(p)
    scipy.io.savemat(df_file, df, oned_as='column')

if eval_function_value:
    f = {}
    f['f'] = cf.f(p)
    scipy.io.savemat(f_file, f, oned_as='column')