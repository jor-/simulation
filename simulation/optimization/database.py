import os.path

import numpy as np

import util.database.array_based
import measurements.all.data
import simulation.model.constants
import simulation.model.options
import simulation.optimization.constants
import simulation.optimization.cost_function


def database_array_file(cost_function_name, model_name,
                        min_standard_deviations=None, min_measurements_correlations=None, max_box_distance_to_water=None):
    measurements_object = measurements.all.data.all_measurements(
        min_standard_deviation=min_standard_deviations,
        min_measurements_correlation=min_measurements_correlations,
        max_box_distance_to_water=max_box_distance_to_water)
    model_options = simulation.model.options.ModelOptions({'model_name': model_name})
    cost_function_class = getattr(simulation.optimization.cost_function, cost_function_name)
    cost_function = cost_function_class(measurements_object, model_options=model_options)

    model_dir = cost_function.model.model_dir
    base_dir = os.path.join(model_dir, 'cf_values')
    filename = 'cf_values.npy'
    array_file = cost_function._filename(filename, base_dir=base_dir)
    return array_file


def update_db(cost_function_name, model_name,
              min_standard_deviations=None, min_measurements_correlations=None, max_box_distance_to_water=None,
              overwrite=True):

    array_file = database_array_file(
        cost_function_name, model_name,
        min_standard_deviations=min_standard_deviations,
        min_measurements_correlations=min_measurements_correlations,
        max_box_distance_to_water=max_box_distance_to_water)
    db = util.database.array_based.Database(array_file, value_length=1)

    measurements_object = measurements.all.data.all_measurements(
        min_standard_deviation=min_standard_deviations,
        min_measurements_correlation=min_measurements_correlations,
        max_box_distance_to_water=max_box_distance_to_water)
    cost_function_class = getattr(simulation.optimization.cost_function, cost_function_name)
    cost_function = cost_function_class(measurements_object)

    for cf in simulation.optimization.cost_function.iterator([cost_function], model_names=[model_name]):
        if cf.f_available():
            concentrations = cf.model.initial_constant_concentrations
            time_step = cf.model.model_options.time_step
            parameters = cf.model.parameters
            key = np.array([*concentrations, time_step, *parameters])
            f_value = cf.f()
            db.set_value_with_key(key, f_value, use_tolerances=False, overwrite=overwrite)


# *** main function for script call *** #

def _main():

    # parse arguments
    import argparse

    parser = argparse.ArgumentParser(description='Update cost function database values.')
    parser.add_argument('cost_function_name', type=str, help='The cost function to evaluate.')
    parser.add_argument('model_name', type=str, choices=simulation.model.constants.MODEL_NAMES, help='The model to evaluate.')
    parser.add_argument('--min_standard_deviations', nargs='+', type=float, default=None, help='The minimal standard deviations assumed for the measurement errors applied for each dataset.')
    parser.add_argument('--min_measurements_correlations', nargs='+', type=int, default=None, help='The minimal number of measurements used to calculate correlations applied to each dataset.')
    parser.add_argument('--max_box_distance_to_water', type=int, default=None, help='The maximal distances to water boxes to accept measurements.')
    parser.add_argument('--debug_level', choices=util.logging.LEVELS, default='INFO', help='Print debug infos low to passed level.')
    parser.add_argument('--version', action='version', version='%(prog)s {}'.format(simulation.__version__))
    args = parser.parse_args()

    # run
    with util.logging.Logger(level=args.debug_level):
        update_db(
            args.cost_function_name, args.model_name,
            min_standard_deviations=args.min_standard_deviations,
            min_measurements_correlations=args.min_measurements_correlations,
            max_box_distance_to_water=args.max_box_distance_to_water,
            overwrite=True)
        util.logging.info('Finished.')


if __name__ == "__main__":
    _main()
