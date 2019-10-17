import numpy as np

import measurements.all.data

import simulation.model.options
import simulation.model.cache
import simulation.optimization.constants
import simulation.accuracy.linearized


def init_model_options(model_name, time_step=1,
                       concentrations=None, concentrations_index=None, parameters=None, parameters_index=None,
                       spinup_years=None, spinup_tolerance=None, spinup_satisfy_years_and_tolerance=True,
                       derivative_years=None, derivative_step_size=None, derivative_accuracy_order=None,
                       parameters_relative_tolerance=None, parameters_absolute_tolerance=None,
                       concentrations_relative_tolerance=None, concentrations_absolute_tolerance=None,
                       concentrations_must_be_set=False, parameters_must_be_set=False):

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
    if parameters_relative_tolerance is not None or parameters_absolute_tolerance is not None:
        parameter_tolerance_options = model_options['parameter_tolerance_options']
        if parameters_relative_tolerance is not None:
            parameter_tolerance_options['relative'] = parameters_relative_tolerance
        if parameters_absolute_tolerance is not None:
            parameter_tolerance_options['absolute'] = parameters_absolute_tolerance

    # set initial concentration tolerance options
    if concentrations_relative_tolerance is not None or concentrations_absolute_tolerance is not None:
        tolerance_options = model_options['initial_concentration_options']['tolerance_options']
        if concentrations_relative_tolerance is not None:
            tolerance_options['relative'] = concentrations_relative_tolerance
        if concentrations_absolute_tolerance is not None:
            tolerance_options['absolute'] = concentrations_absolute_tolerance

    # create model
    if concentrations_index is not None or parameters_index is not None:
        model = simulation.model.cache.Model(model_options=model_options)

    # set initial concentration
    if concentrations is not None:
        c = np.array(concentrations)
    elif concentrations_index is not None:
        c = model._constant_concentrations_db.get_value(concentrations_index)
    if concentrations is not None or concentrations_index is not None:
        model_options.initial_concentration_options.concentrations = c
    elif concentrations_must_be_set:
        raise ValueError('Concentrations or concentrations_index must be set.')

    # set model parameters
    if parameters is not None:
        p = np.array(parameters)
    elif parameters_index is not None:
        p = model._parameters_db.get_value(parameters_index)
    if parameters is not None or parameters_index is not None:
        model_options.parameters = p
    elif parameters_must_be_set:
        raise ValueError('Parameters or parameters_index must be set.')

    return model_options


def init_measurements(model_options,
                      min_measurements_standard_deviation=None, min_standard_deviation=None,
                      min_measurements_correlation=None, correlation_decomposition_min_value_D=None, correlation_decomposition_min_abs_value_L=None,
                      max_box_distance_to_water=None):
    measurements_object = measurements.all.data.all_measurements(
        tracers=model_options.tracers,
        min_measurements_standard_deviation=min_measurements_standard_deviation,
        min_standard_deviation=min_standard_deviation,
        min_measurements_correlation=min_measurements_correlation,
        correlation_decomposition_min_value_D=correlation_decomposition_min_value_D,
        correlation_decomposition_min_abs_value_L=correlation_decomposition_min_abs_value_L,
        max_box_distance_to_water=max_box_distance_to_water,
        water_lsm='TMM',
        sample_lsm='TMM')
    return measurements_object


def argparse_add_model_options(parser):
    parser.add_argument('model_name', choices=simulation.model.constants.MODEL_NAMES, help='The name of the model that should be used.')
    parser.add_argument('--time_step', type=int, default=1, help='The time step of the model that should be used. Default: 1')

    parser.add_argument('--concentrations', type=float, nargs='+', help='The constant concentration values for the tracers in the initial spinup that should be used. If not specified the default model concentrations are used.')
    parser.add_argument('--concentrations_index', type=int, help='The constant concentration index that should be used if no constant concentration values are specified.')

    parser.add_argument('--parameters', type=float, nargs='+', help='The model parameters that should be used.')
    parser.add_argument('--parameters_index', type=int, help='The model parameter index that should be used if no model parameters are specified.')

    parser.add_argument('--spinup_years', type=int, default=10000, help='The number of years for the spinup.')
    parser.add_argument('--spinup_tolerance', type=float, default=0, help='The tolerance for the spinup.')
    parser.add_argument('--spinup_satisfy_years_and_tolerance', action='store_true', help='If used, the spinup is terminated if years and tolerance have been satisfied. Otherwise, the spinup is terminated as soon as years or tolerance have been satisfied.')

    parser.add_argument('--derivative_step_size', type=float, default=None, help='The step size used for the finite difference approximation.')
    parser.add_argument('--derivative_years', type=int, default=None, help='The number of years for the finite difference approximation spinup.')
    parser.add_argument('--derivative_accuracy_order', type=int, default=None, help='The accuracy order used for the finite difference approximation. 1 = forward differences. 2 = central differences.')

    parser.add_argument('--parameters_relative_tolerance', type=float, nargs='+', default=10**-6, help='The relative tolerance up to which two model parameter vectors are considered equal.')
    parser.add_argument('--parameters_absolute_tolerance', type=float, nargs='+', default=10**-6, help='The absolute tolerance up to which two model parameter vectors are considered equal.')

    parser.add_argument('--concentrations_relative_tolerance', type=float, default=10**-6, help='The relative tolerance up to which two initial concentration vectors are considered equal.')
    parser.add_argument('--concentrations_absolute_tolerance', type=float, default=10**-6, help='The absolute tolerance up to which two initial concentration vectors are considered equal.')

    return parser


def argparse_add_measurement_options(parser):
    parser.add_argument('--min_measurements_standard_deviations', nargs='+', type=int, default=None, help='The minimal number of measurements used to calculate standard deviations applied to each dataset.')
    parser.add_argument('--min_standard_deviations', nargs='+', type=float, default=None, help='The minimal standard deviations assumed for the measurement errors applied to each dataset.')
    parser.add_argument('--min_measurements_correlations', nargs='+', type=int, default=None, help='The minimal number of measurements used to calculate correlations applied to each dataset.')
    parser.add_argument('--correlation_decomposition_min_value_D', type=float, default=None, help='The minimal value applied to the diagonal matrix of the decomposition of the correlation matrix applied to each dataset.')
    parser.add_argument('--correlation_decomposition_min_abs_value_L', type=float, default=None, help='The minimal absolute value applied to the lower triangular matrix of the decomposition of the correlation matrix applied to each dataset.')
    parser.add_argument('--max_box_distance_to_water', type=int, default=float('inf'), help='The maximal distance to water boxes to accept measurements.')
    return parser


def argparse_add_accuracy_object_options(parser):
    argparse_add_model_options(parser)
    argparse_add_measurement_options(parser)
    parser.add_argument('--cost_function_name', required=True, choices=simulation.optimization.constants.COST_FUNCTION_NAMES, help='The cost function which should be evaluated.')
    return parser


def parse_model_options(args, concentrations_must_be_set=False, parameters_must_be_set=False):
    model_options = init_model_options(
        args.model_name,
        time_step=args.time_step,
        concentrations=args.concentrations,
        concentrations_index=args.concentrations_index,
        parameters=args.parameters,
        parameters_index=args.parameters_index,
        spinup_years=args.spinup_years,
        spinup_tolerance=args.spinup_tolerance,
        spinup_satisfy_years_and_tolerance=args.spinup_satisfy_years_and_tolerance,
        derivative_years=args.derivative_years,
        derivative_step_size=args.derivative_step_size,
        derivative_accuracy_order=args.derivative_accuracy_order,
        parameters_relative_tolerance=args.parameters_relative_tolerance,
        parameters_absolute_tolerance=args.parameters_absolute_tolerance,
        concentrations_relative_tolerance=args.concentrations_relative_tolerance,
        concentrations_absolute_tolerance=args.concentrations_absolute_tolerance,
        concentrations_must_be_set=concentrations_must_be_set,
        parameters_must_be_set=parameters_must_be_set)
    return model_options


def parse_measurements_options(args, model_options):
    measurements_object = init_measurements(
        model_options,
        min_measurements_standard_deviation=args.min_measurements_standard_deviations,
        min_standard_deviation=args.min_standard_deviations,
        min_measurements_correlation=args.min_measurements_correlations,
        correlation_decomposition_min_value_D=args.correlation_decomposition_min_value_D,
        correlation_decomposition_min_abs_value_L=args.correlation_decomposition_min_abs_value_L,
        max_box_distance_to_water=args.max_box_distance_to_water)
    return measurements_object


def parse_accuracy_object_options(args, concentrations_must_be_set=False, parameters_must_be_set=False):
    model_options = parse_model_options(args, concentrations_must_be_set=concentrations_must_be_set, parameters_must_be_set=concentrations_must_be_set)
    measurements_object = parse_measurements_options(args, model_options)
    cost_function_name = args.cost_function_name
    try:
        accuracy_class = getattr(simulation.accuracy.linearized, cost_function_name)
    except AttributeError:
        raise ValueError('Unknown accuracy class {}.'.format(cost_function_name))
    accuracy_object = accuracy_class(measurements_object=measurements_object, model_options=model_options)
    return accuracy_object
