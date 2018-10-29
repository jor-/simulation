import numpy as np

import util.multi_dict

import simulation.model.options
import simulation.optimization.cost_function

import util.logging


def cost_functions_for_all_measurements(max_box_distance_to_water=None, min_standard_deviations=None, min_measurements_correlations=None, cost_function_classes=None):
    model_options = simulation.model.options.ModelOptions()
    model_options.spinup_options = {'years': 1, 'tolerance': 0.0, 'combination': 'or'}
    cost_functions = simulation.optimization.cost_function.cost_functions_for_all_measurements(
        min_standard_deviations=min_standard_deviations,
        max_box_distance_to_water=max_box_distance_to_water,
        min_measurements_correlations=min_measurements_correlations,
        cost_function_classes=cost_function_classes,
        model_options=model_options)
    return cost_functions


def all_values(cost_functions, model_names=None):
    all_values_dict = util.multi_dict.MultiDict(sorted=True)

    for cost_function in simulation.optimization.cost_function.iterator(cost_functions, model_names=model_names):
        model_options = cost_function.model.model_options
        key = (model_options.model_name,
               str(cost_function),
               model_options.time_step,
               model_options.initial_concentration_options.concentrations,
               model_options.parameters)
        if cost_function.f_available():
            value = cost_function.f()
        else:
            value = np.nan
        all_values_dict.append_value(key, value)

    return all_values_dict


def all_values_for_all_measurements(max_box_distance_to_water=None, min_standard_deviations=None, min_measurements_correlations=None, cost_function_classes=None, model_names=None):
    cost_functions = cost_functions_for_all_measurements(
        max_box_distance_to_water=max_box_distance_to_water,
        min_standard_deviations=min_standard_deviations,
        min_measurements_correlations=min_measurements_correlations,
        cost_function_classes=cost_function_classes)
    all_values_dict = all_values(cost_functions, model_names)
    return all_values_dict


# *** main function for script call *** #

def _main():

    # parse arguments
    import argparse

    parser = argparse.ArgumentParser(description='Getting all cost function values.')
    parser.add_argument('--min_standard_deviations', nargs='+', type=float, default=None, help='The minimal standard deviations assumed for the measurement errors applied for each dataset.')
    parser.add_argument('--min_measurements_correlations', nargs='+', type=int, default=None, help='The minimal number of measurements used to calculate correlations applied to each dataset.')
    parser.add_argument('--max_box_distance_to_water', type=int, default=None, help='The maximal distances to water boxes to accept measurements.')
    parser.add_argument('--cost_functions', type=str, default=None, nargs='+', help='The cost functions to evaluate.')
    parser.add_argument('--model_names', type=str, default=None, choices=simulation.model.constants.MODEL_NAMES, nargs='+', help='The models to evaluate.')
    parser.add_argument('--save_file', type=str, default=None, help='A file where to save the result.')
    parser.add_argument('-d', '--debug_level', choices=util.logging.LEVELS, default='INFO', help='Print debug infos low to passed level.')
    args = parser.parse_args()

    # cost_function_classes
    if args.cost_functions is None:
        cost_function_classes = None
    else:
        cost_function_classes = [getattr(simulation.optimization.cost_function, cost_function_name) for cost_function_name in args.cost_functions]

    # run
    with util.logging.Logger(level=args.debug_level):
        results = all_values_for_all_measurements(
            max_box_distance_to_water=args.max_box_distance_to_water,
            min_standard_deviations=args.min_standard_deviations,
            min_measurements_correlations=args.min_measurements_correlations,
            cost_function_classes=cost_function_classes,
            model_names=args.model_names)
        util.logging.info(str(results))
        if args.save_file is not None:
            results.save(args.save_file)
        util.logging.info('Finished.')


if __name__ == "__main__":
    _main()
