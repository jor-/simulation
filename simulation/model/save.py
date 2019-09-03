import numpy as np

import simulation
import simulation.model.cache
import simulation.model.options
import simulation.model.constants

import measurements.all.data

import util.logging


def prepare_model_options(model_name, time_step=1, concentrations=None, concentrations_index=None, parameters=None, parameter_set_index=None,
                          spinup_years=None, spinup_tolerance=None, spinup_satisfy_years_and_tolerance=True,
                          derivative_years=None, derivative_step_size=None, derivative_accuracy_order=None,
                          model_parameters_relative_tolerance=None, model_parameters_absolute_tolerance=None,
                          initial_concentrations_relative_tolerance=None, initial_concentrations_absolute_tolerance=None):

    # prepare model options
    model_options = simulation.model.options.ModelOptions()
    model_options.model_name = model_name
    model_options.time_step = time_step

    # set spinup options
    if spinup_years is not None and spinup_tolerance is None:
        spinup_tolerance = 0
        spinup_satisfy_years_and_tolerance = False
    if spinup_tolerance is not None and spinup_years is None:
        spinup_years = 10**10
        spinup_satisfy_years_and_tolerance = False
    if spinup_years is not None and spinup_tolerance is not None:
        if spinup_satisfy_years_and_tolerance:
            combination = 'and'
        else:
            combination = 'or'
        spinup_options = {'years': spinup_years, 'tolerance': spinup_tolerance, 'combination': combination}
        model_options.spinup_options = spinup_options

    # set derivative options
    derivative_options = {}
    if derivative_step_size is not None:
        derivative_options['step_size'] = derivative_step_size
    if derivative_years is not None:
        derivative_options['years'] = derivative_years
    if derivative_accuracy_order is not None:
        derivative_options['accuracy_order'] = derivative_accuracy_order
    if len(derivative_options) > 0:
        model_options.derivative_options = derivative_options

    # set model parameters tolerance options
    if model_parameters_relative_tolerance is not None or model_parameters_absolute_tolerance is not None:
        parameter_tolerance_options = model_options['parameter_tolerance_options']
        if model_parameters_relative_tolerance is not None:
            parameter_tolerance_options['relative'] = model_parameters_relative_tolerance
        if model_parameters_absolute_tolerance is not None:
            parameter_tolerance_options['absolute'] = model_parameters_absolute_tolerance

    # set initial concentration tolerance options
    if initial_concentrations_relative_tolerance is not None or initial_concentrations_absolute_tolerance is not None:
        tolerance_options = model_options['initial_concentration_options']['tolerance_options']
        if initial_concentrations_relative_tolerance is not None:
            tolerance_options['relative'] = initial_concentrations_relative_tolerance
        if initial_concentrations_absolute_tolerance is not None:
            tolerance_options['absolute'] = initial_concentrations_absolute_tolerance

    # create model
    model = simulation.model.cache.Model(model_options=model_options)

    # set initial concentration
    if concentrations is not None:
        c = np.array(concentrations)
    elif concentrations_index is not None:
        c = model._constant_concentrations_db.get_value(concentrations_index)
    if concentrations is not None or concentrations_index is not None:
        model_options.initial_concentration_options.concentrations = c

    # set model parameters
    if parameters is not None:
        p = np.array(parameters)
    elif parameter_set_index is not None:
        p = model._parameters_db.get_value(parameter_set_index)
    if parameters is not None or parameter_set_index is not None:
        model_options.parameters = p

    return model_options


def prepare_measurements(model_options,
                         min_measurements_standard_deviations=None, min_standard_deviations=None, min_measurements_correlations=None, min_diag_correlations=None, max_box_distance_to_water=None):
    measurements_object = measurements.all.data.all_measurements(
        tracers=model_options.tracers,
        min_measurements_standard_deviation=min_measurements_standard_deviations,
        min_standard_deviation=min_standard_deviations,
        min_measurements_correlation=min_measurements_correlations,
        min_diag_correlations=min_diag_correlations,
        max_box_distance_to_water=max_box_distance_to_water,
        water_lsm='TMM',
        sample_lsm='TMM')
    return measurements_object


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

    parser.add_argument('model_name', choices=simulation.model.constants.MODEL_NAMES, help='The name of the model that should be used.')
    parser.add_argument('--time_step', type=int, default=1, help='The time step of the model that should be used. Default: 1')

    parser.add_argument('--concentrations', type=float, nargs='+', help='The constant concentration values for the tracers in the initial spinup that should be used. If not specified the default model concentrations are used.')
    parser.add_argument('--concentrations_index', type=int, help='The constant concentration index that should be used if no constant concentration values are specified.')

    parser.add_argument('--parameters', type=float, nargs='+', help='The model parameters that should be used.')
    parser.add_argument('--parameter_set_index', type=int, help='The model parameter index that should be used if no model parameters are specified.')

    parser.add_argument('--spinup_years', type=int, default=10000, help='The number of years for the spinup.')
    parser.add_argument('--spinup_tolerance', type=float, default=0, help='The tolerance for the spinup.')
    parser.add_argument('--spinup_satisfy_years_and_tolerance', action='store_true', help='If used, the spinup is terminated if years and tolerance have been satisfied. Otherwise, the spinup is terminated as soon as years or tolerance have been satisfied.')

    parser.add_argument('--derivative_step_size', type=float, default=None, help='The step size used for the finite difference approximation.')
    parser.add_argument('--derivative_years', type=int, default=None, help='The number of years for the finite difference approximation spinup.')
    parser.add_argument('--derivative_accuracy_order', type=int, default=None, help='The accuracy order used for the finite difference approximation. 1 = forward differences. 2 = central differences.')

    parser.add_argument('--min_measurements_standard_deviations', nargs='+', type=int, default=None, help='The minimal number of measurements used to calculate standard deviations applied to each dataset.')
    parser.add_argument('--min_standard_deviations', nargs='+', type=float, default=None, help='The minimal standard deviations assumed for the measurement errors applied to each dataset.')
    parser.add_argument('--min_measurements_correlations', nargs='+', type=int, default=None, help='The minimal number of measurements used to calculate correlations applied to each dataset.')
    parser.add_argument('--min_diag_correlations', type=float, default=None, help='The minimal value aplied to the diagonal of the decomposition of the correlation matrix applied to each dataset.')
    parser.add_argument('--max_box_distance_to_water', type=int, default=float('inf'), help='The maximal distance to water boxes to accept measurements.')

    parser.add_argument('--eval_function', '-f', action='store_true', help='Save the value of the model.')
    parser.add_argument('--eval_first_derivative', '-df', action='store_true', help='Save the values of the derivative of the model.')
    parser.add_argument('--eval_second_derivative', '-d2f', action='store_true', help='Save the values of the second derivative of the model.')

    parser.add_argument('--all_values_time_dim', type=int, help='Set time dim for box values. If None, eval measurement values.')

    parser.add_argument('-d', '--debug', action='store_true', help='Print debug infos.')

    parser.add_argument('--version', action='version', version='%(prog)s {}'.format(simulation.__version__))

    args = parser.parse_args()

    if args.concentrations is None and args.concentrations_index is None:
        raise ValueError('"--concentrations" or "--concentrations_index" must be specified. Use "--help" for more infos.')
    if args.parameters is None and args.parameter_set_index is None:
        raise ValueError('"--concentrations" or "--concentrations_index" must be specified. Use "--help" for more infos.')

    # call function
    with util.logging.Logger():

        model_options = prepare_model_options(
            args.model_name,
            time_step=args.time_step,
            spinup_years=args.spinup_years,
            spinup_tolerance=args.spinup_tolerance,
            spinup_satisfy_years_and_tolerance=args.spinup_satisfy_years_and_tolerance,
            concentrations=args.concentrations,
            concentrations_index=args.concentrations_index,
            parameters=args.parameters,
            parameter_set_index=args.parameter_set_index,
            derivative_years=args.derivative_years,
            derivative_step_size=args.derivative_step_size,
            derivative_accuracy_order=args.derivative_accuracy_order)

        measurements_object = prepare_measurements(
            model_options,
            min_measurements_standard_deviations=args.min_measurements_standard_deviations,
            min_standard_deviations=args.min_standard_deviations,
            min_measurements_correlations=args.min_measurements_correlations,
            min_diag_correlations=args.min_diag_correlations,
            max_box_distance_to_water=args.max_box_distance_to_water)

        save(model_options, measurements_object,
             eval_function=args.eval_function,
             eval_first_derivative=args.eval_first_derivative,
             eval_second_derivative=args.eval_second_derivative,
             all_values_time_dim=args.all_values_time_dim,
             debug_output=args.debug)


if __name__ == "__main__":
    _main()
