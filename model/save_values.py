import argparse
import os.path
import numpy as np

import ndop.util.data_base

import util.batch.universal.system
import util.logging

from ndop.model.constants import JOB_MEMORY_GB


if __name__ == "__main__":
    ## configure arguments
    parser = argparse.ArgumentParser(description='Starting model spinup.')
    parser.add_argument('--p', type=float, nargs=7)
    parser.add_argument('--parameter_set', type=int)
    parser.add_argument('--years', type=int, default=10000)
    parser.add_argument('--tolerance', type=float, default=0.0)
    parser.add_argument('--combination', choices=['or', 'and'], default='or')
    parser.add_argument('--time_dim', type=int, default=12)
    parser.add_argument('--F', action='store_true', help='Eval F.')
    parser.add_argument('--DF', action='store_true', help='Eval DF.')
    parser.add_argument('-d', '--debug', action='store_true', help='Print debug infos.')
    parser.add_argument('--version', action='version', version='%(prog)s 0.1')
    args = vars(parser.parse_args())
    
    ## get args
    years = args['years']
    tolerance = args['tolerance']
    combination = args['combination']
    eval_F = args['F']
    eval_DF = args['DF']
    time_dim = args['time_dim']
    
    
    ## get p
    p = args['p']
    if p is not None:
        p = np.array(p)
    else:
        parameter_set_number = args['parameter_set']
        from ndop.model.constants import MODEL_OUTPUT_DIR, MODEL_TIME_STEP_DIRNAME, MODEL_PARAMETERS_SET_DIRNAME, MODEL_PARAMETERS_FILENAME
        time_step = 1
        parameters_file = os.path.join(MODEL_OUTPUT_DIR, MODEL_TIME_STEP_DIRNAME.format(time_step), MODEL_PARAMETERS_SET_DIRNAME.format(parameter_set_number), MODEL_PARAMETERS_FILENAME)
        p = np.loadtxt(parameters_file)
        
        
    ## run
    with util.logging.Logger(disp_stdout=args['debug']):
        from ndop.optimization.constants import COST_FUNCTION_NODES_SETUP_SPINUP, COST_FUNCTION_NODES_SETUP_DERIVATIVE, COST_FUNCTION_NODES_SETUP_TRAJECTORY
        job_setup = {'name':'NDOP'}
        job_setup['spinup'] = {'nodes_setup' : COST_FUNCTION_NODES_SETUP_SPINUP}
        job_setup['derivative'] = {'nodes_setup' : COST_FUNCTION_NODES_SETUP_DERIVATIVE}
        job_setup['trajectory'] = {'nodes_setup' : COST_FUNCTION_NODES_SETUP_TRAJECTORY}
        spinup_setup = {'years':years, 'tolerance':tolerance, 'combination':combination}
        db = ndop.util.data_base.DataBase(spinup_setup, job_setup=job_setup)
        if eval_F:
            db.f_boxes(p, time_dim)
        if eval_DF:
            db.df_boxes(p, time_dim)
#         model = ndop.model.eval.Model(job_setup)
#         if eval_F:
#             model.f_boxes(p, time_dim, spinup_setup)
#         if eval_DF:
#             model.df_boxes(p, time_dim, spinup_setup)
            