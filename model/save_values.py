import argparse
import os.path
import numpy as np

import simulation.util.data_base

import util.batch.universal.system
import util.logging


if __name__ == "__main__":
    ## configure arguments
    parser = argparse.ArgumentParser(description='Starting model spinup.')
    parser.add_argument('--p', type=float, nargs=7)
    parser.add_argument('--parameter_set_index', type=int)

    parser.add_argument('--eval_function_value', '-f', action='store_true', help='Save the value of the cost function.')
    parser.add_argument('--eval_grad_value', '-g', action='store_true', help='Save the values of the derivative of the cost function.')
    
    parser.add_argument('--spinup_years', '-y', type=int, default=10000, help='The number of years for the spinup.')
    parser.add_argument('--spinup_tolerance', '-t', type=float, default=0, help='The tolerance for the spinup.')
    parser.add_argument('--spinup_satisfy_years_and_tolerance', '-a', action='store_true', help='If used, the spinup is terminated if years and tolerance have been satisfied. Otherwise, the spinup is terminated as soon as years or tolerance have been satisfied.')

    parser.add_argument('--derivative_step_size', type=float, default=None, help='The step size used for the finite difference approximation.')
    parser.add_argument('--derivative_years', type=int, default=None, help='The number of years for the finite difference approximation spinup.')
    parser.add_argument('--derivative_accuracy_order', type=int, default=None, help='The accuracy order used for the finite difference approximation. 1 = forward differences. 2 = central differences.')
    
    parser.add_argument('--time_dim', type=int, default=12)
    parser.add_argument('-d', '--debug', action='store_true', help='Print debug infos.')
    parser.add_argument('--version', action='version', version='%(prog)s 0.1')

    args = parser.parse_args()

    ## check wich values to evaluate
    eval_function_value = args.eval_function_value
    eval_grad_value = args.eval_grad_value

    ## prepare spinup options
    if args.spinup_satisfy_years_and_tolerance:
        combination='and'
    else:
        combination='or'
    spinup_options = {'years': args.spinup_years, 'tolerance': args.spinup_tolerance, 'combination':combination}

    ## prepare derivative options
    derivative_options = {}
    if args.derivative_step_size is not None:
        derivative_options['step_size'] = args.derivative_step_size
    if args.derivative_years is not None:
        derivative_options['years'] = args.derivative_years
    if args.derivative_accuracy_order is not None:
        derivative_options['accuracy_order'] = args.derivative_accuracy_order
        
    ## prepare model options
    model_options = {}
    model_options['spinup_options'] = spinup_options
    model_options['derivative_options'] = derivative_options
    
    ## prepare job option
    from simulation.optimization.constants import COST_FUNCTION_NODES_SETUP_SPINUP, COST_FUNCTION_NODES_SETUP_DERIVATIVE, COST_FUNCTION_NODES_SETUP_TRAJECTORY
    job_setup = {'name':'NDOP'}
    job_setup['spinup'] = {'nodes_setup' : COST_FUNCTION_NODES_SETUP_SPINUP}
    job_setup['derivative'] = {'nodes_setup' : COST_FUNCTION_NODES_SETUP_DERIVATIVE}
    job_setup['trajectory'] = {'nodes_setup' : COST_FUNCTION_NODES_SETUP_TRAJECTORY}

    ## create model
    with util.logging.Logger(disp_stdout=args.debug):
        db = simulation.util.data_base.DataBase(model_options=model_options, job_setup=job_setup)
        
        ## get p
        if args.p is not None:
            p = np.array(args.p)
        else:
            p = db.model._parameter_db.get_value(args.parameter_set_index)
        
        ## eval
        if args.eval_function_value:
            db.f_boxes(p, args.time_dim)
        if args.eval_grad_value:
            db.df_boxes(p, args.time_dim)
        # db = simulation.util.data_base.DataBase(spinup_options=spinup_options, derivative_options=derivative_options, job_setup=job_setup)
        # if args.eval_function_value:
        #     db.f_boxes(p, args.time_dim)
        # if args.eval_grad_value:
        #     db.df_boxes(p, args.time_dim)
