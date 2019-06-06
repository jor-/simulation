import os

import util.database.array_based

import measurements.all.data

import simulation.model.constants
import simulation.optimization.cost_function


def database_for_cost_function(cost_function):
    # get array file
    model_dir = cost_function.model.model_dir
    base_dir = os.path.join(model_dir, 'cf_values')
    filename = 'cf_values.npy'
    array_file = cost_function._filename(filename, base_dir=base_dir)
    array_file = array_file.replace(os.sep + simulation.model.constants.DATABASE_CACHE_SPINUP_DIRNAME, '')
    # init database
    db = util.database.array_based.Database(array_file, value_length=1)
    return db


def update_db(cost_function_name, model_name,
              min_standard_deviations=None, min_measurements_standard_deviations=None, min_measurements_correlations=None, max_box_distance_to_water=None,
              overwrite=True):

    measurements_object = measurements.all.data.all_measurements(
        min_standard_deviation=min_standard_deviations,
        min_measurements_standard_deviation=min_measurements_standard_deviations,
        min_measurements_correlation=min_measurements_correlations,
        max_box_distance_to_water=max_box_distance_to_water,
        water_lsm='TMM',
        sample_lsm='TMM')
    cost_function_class = getattr(simulation.optimization.cost_function, cost_function_name)
    cost_function = cost_function_class(measurements_object, use_global_value_database=True)

    for cf in simulation.optimization.cost_function.iterator([cost_function], model_names=[model_name]):
        if cf.f_available():
            value = cf.f()
            cf._add_value_to_database(value, overwrite=overwrite)


# *** main function for script call *** #

def _main():

    # parse arguments
    import argparse

    parser = argparse.ArgumentParser(description='Update cost function database values.')
    parser.add_argument('cost_function_name', type=str, help='The cost function to evaluate.')
    parser.add_argument('model_name', type=str, choices=simulation.model.constants.MODEL_NAMES, help='The model to evaluate.')
    parser.add_argument('--min_standard_deviations', nargs='+', type=float, default=None, help='The minimal standard deviations assumed for the measurement errors applied for each dataset.')
    parser.add_argument('--min_measurements_standard_deviations', nargs='+', type=int, default=None, help='The minimal number of measurements used to calculate standard deviations applied to each dataset.')
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
            min_measurements_standard_deviations=args.min_measurements_standard_deviations,
            min_measurements_correlations=args.min_measurements_correlations,
            max_box_distance_to_water=args.max_box_distance_to_water,
            overwrite=True)
        util.logging.info('Finished.')


if __name__ == "__main__":
    _main()
