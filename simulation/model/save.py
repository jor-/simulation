import argparse

import numpy as np

import simulation.model.cache
import simulation.model.options
import simulation.model.constants

import measurements.all.pw.data

import util.logging
logger = util.logging.get_logger()



def save(model_name, time_step=1, spinup_years=10000, spinup_tolerance=0, spinup_satisfy_years_and_tolerance=False, concentrations=None, concentrations_index=None, parameters=None, parameter_set_index=None, derivative_years=None, derivative_step_size=None, derivative_accuracy_order=None, eval_function_value=True, eval_grad_value=True, all_values_time_dim=None, debug_output=True):

    ## prepare model options
    model_options = simulation.model.options.ModelOptions()
    model_options.model_name = model_name
    model_options.time_step = time_step

    ## set spinup options
    if spinup_satisfy_years_and_tolerance:
        combination='and'
    else:
        combination='or'
    spinup_options = {'years': spinup_years, 'tolerance': spinup_tolerance, 'combination':combination}
    model_options.spinup_options = spinup_options

    ## set derivative options
    derivative_options = {}
    if derivative_step_size is not None:
        derivative_options['step_size'] = derivative_step_size
    if derivative_years is not None:
        derivative_options['years'] = derivative_years
    if derivative_accuracy_order is not None:
        derivative_options['accuracy_order'] = derivative_accuracy_order
    model_options.derivative_options = derivative_options
        
    ## prepare job option
    from simulation.optimization.constants import COST_FUNCTION_NODES_SETUP_SPINUP, COST_FUNCTION_NODES_SETUP_DERIVATIVE, COST_FUNCTION_NODES_SETUP_TRAJECTORY
    job_options = {'name':'NDOP'}
    job_options['spinup'] = {'nodes_setup' : COST_FUNCTION_NODES_SETUP_SPINUP}
    job_options['derivative'] = {'nodes_setup' : COST_FUNCTION_NODES_SETUP_DERIVATIVE}
    job_options['trajectory'] = {'nodes_setup' : COST_FUNCTION_NODES_SETUP_TRAJECTORY}

    ## create model
    with util.logging.Logger(disp_stdout=debug_output):
        model = simulation.model.cache.Model(model_options=model_options, job_options=job_options)
    
        ## set initial concentration
        if concentrations is not None:
            c = np.array(concentrations)
        else:
            c = model._constant_concentrations_db.get_value(concentrations_index)
        model_options.initial_concentration_options.concentrations = c
    
        ## set model parameters
        if parameters is not None:
            p = np.array(parameters)
        else:
            p = model._parameter_db.get_value(parameter_set_index)
        model_options.parameters = p
        
        ## eval all box values
        if all_values_time_dim is not None:
            if eval_function_value:
                model.f_all(all_values_time_dim)
            if eval_grad_value:
                model.df_all(all_values_time_dim)
        ## eval measurement values
        else:
            if eval_function_value:
                model.f_measurements(*measurements.all.pw.data.all_measurements())
            if eval_grad_value:
                model.df_measurements(*measurements.all.pw.data.all_measurements())


def save_all(concentration_indices=None, time_steps=None, parameter_set_indices=None):
    if time_steps is None:
        time_steps = simulation.model.constants.METOS_TIME_STEPS
    use_fix_parameter_sets = parameter_set_indices is not None
    measurements_list = measurements.all.pw.data.all_measurements()

    model = simulation.model.cache.Model()
    
    model_options = model.model_options
    model_options.spinup_options.years = 1
    model_options.spinup_options.tolerance = 0
    model_options.spinup_options.combination = 'or'
    
    for model_name in simulation.model.constants.MODEL_NAMES:
        model_options.model_name = model_name
        for concentration_db in (model._constant_concentrations_db, model._vector_concentrations_db):
            if concentration_indices is None:
                concentration_indices = concentration_db.used_indices()
            for concentration_index in concentration_indices:
                model_options.initial_concentration_options.concentrations = concentration_db.get_value(concentration_index)
                for time_step in time_steps:
                    model_options.time_step = time_step
                    if not use_fix_parameter_sets:
                        parameter_set_indices = model._parameter_db.used_indices()
                    for parameter_set_index in parameter_set_indices:
                        model_options.parameters = model._parameter_db.get_value(parameter_set_index)
                        logger.info('Calculating model output in {}.'.format(model.parameter_set_dir))
                        model.f_measurements(*measurements_list)
                


if __name__ == "__main__":
    ## configure arguments
    parser = argparse.ArgumentParser(description='Evaluate and save model values.')
    
    parser.add_argument('--model_name', default=simulation.model.constants.MODEL_NAMES[0], choices=simulation.model.constants.MODEL_NAMES, help='The name of the model that should be used.')
    parser.add_argument('--time_step', type=int, default=1, help='The time step of the model that should be used. Default: 1')
     
    parser.add_argument('--concentrations', type=float, nargs='+', help='The constant concentration values for the tracers in the initial spinup that should be used. If not specified the default model concentrations are used.')
    parser.add_argument('--concentrations_index', type=int, help='The constant concentration index that should be used if no constant concentration values are specified.')
    
    parser.add_argument('--parameters', type=float, nargs='+', help='The model parameters that should be used.')
    parser.add_argument('--parameter_set_index', type=int, help='The model parameter index that should be used if no model parameters are specified.')

    parser.add_argument('--eval_function_value', '-f', action='store_true', help='Save the value of the model.')
    parser.add_argument('--eval_grad_value', '-df', action='store_true', help='Save the values of the derivative of the model.')
    
    parser.add_argument('--spinup_years', type=int, default=10000, help='The number of years for the spinup.')
    parser.add_argument('--spinup_tolerance', type=float, default=0, help='The tolerance for the spinup.')
    parser.add_argument('--spinup_satisfy_years_and_tolerance', action='store_true', help='If used, the spinup is terminated if years and tolerance have been satisfied. Otherwise, the spinup is terminated as soon as years or tolerance have been satisfied.')

    parser.add_argument('--derivative_step_size', type=float, default=None, help='The step size used for the finite difference approximation.')
    parser.add_argument('--derivative_years', type=int, default=None, help='The number of years for the finite difference approximation spinup.')
    parser.add_argument('--derivative_accuracy_order', type=int, default=None, help='The accuracy order used for the finite difference approximation. 1 = forward differences. 2 = central differences.')
    
    parser.add_argument('--all_values_time_dim', type=int, help='Set time dim for box values. If None, eval measurement values.')
    
    parser.add_argument('-d', '--debug', action='store_true', help='Print debug infos.')
    parser.add_argument('--version', action='version', version='%(prog)s 0.1')

    args = parser.parse_args()
    
    save(args.model_name, time_step=args.time_step, spinup_years=args.spinup_years, spinup_tolerance=args.spinup_tolerance, spinup_satisfy_years_and_tolerance=args.spinup_satisfy_years_and_tolerance, concentrations=args.concentrations, concentrations_index=args.concentrations_index, parameters=args.parameters, parameter_set_index=args.parameter_set_index, derivative_years=args.derivative_years, derivative_step_size=args.derivative_step_size, derivative_accuracy_order=args.derivative_accuracy_order, eval_function_value=args.eval_function_value, eval_grad_value=args.eval_grad_value, all_values_time_dim=args.all_values_time_dim, debug_output=args.debug)


