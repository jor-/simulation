import numpy as np

import simulation
import simulation.model.cache
import simulation.model.options
import simulation.model.constants
import simulation.util.args

import measurements.all.data

import util.logging


def save(model_options, measurements_object,
         debug_output=True, eval_function=True, eval_first_derivative=True, eval_second_derivative=True, all_values_time_dim=None):

    # prepare job option
    job_options = {'name': 'NDOP'}
    job_options['spinup'] = {'nodes_setup': simulation.model.constants.NODES_SETUP_SPINUP}
    job_options['derivative'] = {'nodes_setup': simulation.model.constants.NODES_SETUP_DERIVATIVE}
    job_options['trajectory'] = {'nodes_setup': simulation.model.constants.NODES_SETUP_TRAJECTORY}

    # create model
    with util.logging.Logger(disp_stdout=debug_output):
        model = simulation.model.cache.Model(model_options=model_options, job_options=job_options)

        # eval all box values
        if all_values_time_dim is not None:
            if eval_function:
                model.f_all(all_values_time_dim)
            if eval_first_derivative:
                model.df_all(all_values_time_dim)
        # eval measurement values
        else:
            if eval_function:
                model.f_measurements(*measurements_object)
            if eval_first_derivative:
                model.df_measurements(*measurements_object, derivative_order=1)
            if eval_second_derivative:
                model.df_measurements(*measurements_object, derivative_order=2)


def save_all(concentration_indices=None, time_steps=None, parameter_set_indices=None):
    if time_steps is None:
        time_steps = simulation.model.constants.METOS_TIME_STEPS
    use_fix_parameter_sets = parameter_set_indices is not None
    measurements_list = measurements.all.data.all_measurements()

    model = simulation.model.cache.Model()

    model_options = model.model_options
    model_options.spinup_options.years = 1
    model_options.spinup_options.tolerance = 0
    model_options.spinup_options.combination = 'or'

    for model_name in simulation.model.constants.MODEL_NAMES:
        model_options.model_name = model_name
        for concentration_db in (model._constant_concentrations_db, model._vector_concentrations_db):
            if concentration_indices is None:
                concentration_indices = concentration_db.all_indices()
            for concentration_index in concentration_indices:
                model_options.initial_concentration_options.concentrations = concentration_db.get_value(concentration_index)
                for time_step in time_steps:
                    model_options.time_step = time_step
                    if not use_fix_parameter_sets:
                        parameter_set_indices = model._parameters_db.all_indices()
                    for parameter_set_index in parameter_set_indices:
                        model_options.parameters = model._parameters_db.get_value(parameter_set_index)
                        util.logging.info('Calculating model output in {}.'.format(model.parameter_set_dir))
                        model.f_measurements(*measurements_list)


# *** main function for script call *** #

def _main():

    # parse arguments
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate and save model values.')

    simulation.util.args.argparse_add_model_options(parser)
    simulation.util.args.argparse_add_measurement_options(parser)

    parser.add_argument('--eval_function', '-f', action='store_true', help='Save the value of the model.')
    parser.add_argument('--eval_first_derivative', '-df', action='store_true', help='Save the values of the derivative of the model.')
    parser.add_argument('--eval_second_derivative', '-d2f', action='store_true', help='Save the values of the second derivative of the model.')

    parser.add_argument('--all_values_time_dim', type=int, help='Set time dim for box values. If None, eval measurement values.')

    parser.add_argument('-d', '--debug', action='store_true', help='Print debug infos.')

    parser.add_argument('--version', action='version', version='%(prog)s {}'.format(simulation.__version__))

    args = parser.parse_args()

    if args.concentrations is None and args.concentrations_index is None:
        raise ValueError('"--concentrations" or "--concentrations_index" must be specified. Use "--help" for more infos.')
    if args.parameters is None and args.parameters_index is None:
        raise ValueError('"--concentrations" or "--concentrations_index" must be specified. Use "--help" for more infos.')

    # call function
    with util.logging.Logger():
        model_options = simulation.util.args.parse_model_options(args)
        measurements_object = simulation.util.args.parse_measurements_options(args, model_options)
        save(model_options, measurements_object,
             eval_function=args.eval_function,
             eval_first_derivative=args.eval_first_derivative,
             eval_second_derivative=args.eval_second_derivative,
             all_values_time_dim=args.all_values_time_dim,
             debug_output=args.debug)


if __name__ == "__main__":
    _main()
