import numpy as np

import simulation
import simulation.optimization.cost_function
import simulation.optimization.job
import simulation.model.constants
import simulation.model.options

import util.batch.universal.system
import util.logging


def save(cost_functions, model_names=None, eval_f=True, eval_df=False):
    for cost_function in simulation.optimization.cost_function.iterator(cost_functions, model_names=model_names):
        if eval_f and not cost_function.f_available():
            try:
                cost_function.f()
            except Exception:
                util.logging.error('Model function could not be evaluated.', exc_info=True)
            else:
                util.logging.info('Saving cost function {} f value in {}'.format(cost_function, cost_function.model.parameter_set_dir))
        if eval_df and not cost_function.df_available():
            try:
                cost_function.df()
            except Exception:
                util.logging.error('Model function derivative could not be evaluated.', exc_info=True)
            else:
                util.logging.info('Saving cost function {} df value in {}'.format(cost_function, cost_function.model.parameter_set_dir))


def save_for_all_measurements(max_box_distance_to_water_list=None, min_standard_deviation_list=None, min_measurements_correlation_list=None, cost_function_classes=None, model_names=None, eval_f=True, eval_df=False):
    model_options = simulation.model.options.ModelOptions()
    model_options.spinup_options = {'years': 1, 'tolerance': 0.0, 'combination': 'or'}
    cost_functions = simulation.optimization.cost_function.cost_functions_for_all_measurements(max_box_distance_to_water_list=max_box_distance_to_water_list, min_standard_deviation_list=min_standard_deviation_list, min_measurements_correlation_list=min_measurements_correlation_list, cost_function_classes=cost_function_classes, model_options=model_options)
    save(cost_functions, model_names=model_names, eval_f=eval_f, eval_df=eval_df)


# *** main function for script call *** #

def _main():

    # parse arguments
    import argparse

    parser = argparse.ArgumentParser(description='Calculating cost function values.')
    parser.add_argument('--min_measurements_correlation_list', type=int, default=None, nargs='+', help='The minimal number of measurements used to calculate correlations.')
    parser.add_argument('--min_standard_deviation_list', type=float, default=None, nargs='+', help='The minimal standard deviation of measurements.')
    parser.add_argument('--max_box_distance_to_water_list', type=int, default=None, nargs='+', help='The maximal distances to water boxes to accept measurements.')
    parser.add_argument('--cost_function_list', type=str, default=None, nargs='+', help='The cost function to evaluate.')
    parser.add_argument('--model_list', type=str, default=None, choices=simulation.model.constants.MODEL_NAMES, nargs='+', help='The models to evaluate.')
    parser.add_argument('--DF', action='store_true', help='Eval (also) DF.')
    parser.add_argument('--debug_level', choices=util.logging.LEVELS, default='INFO', help='Print debug infos low to passed level.')
    parser.add_argument('--version', action='version', version='%(prog)s {}'.format(simulation.__version__))
    args = parser.parse_args()

    # max_box_distance_to_water
    if args.max_box_distance_to_water_list is not None:
        max_box_distance_to_water = np.array(args.max_box_distance_to_water_list)
        add_inf = np.any(max_box_distance_to_water < 0)
        max_box_distance_to_water = max_box_distance_to_water[max_box_distance_to_water >= 0]
        max_box_distance_to_water = np.unique(max_box_distance_to_water)
        max_box_distance_to_water_list = max_box_distance_to_water.tolist()
        if add_inf:
            max_box_distance_to_water_list = max_box_distance_to_water_list + [float('inf')]
    else:
        max_box_distance_to_water_list = [float('inf')]
    # cost_function_classes
    if args.cost_function_list is None:
        cost_function_classes = None
    else:
        cost_function_classes = [getattr(simulation.optimization.cost_function, cost_function_name) for cost_function_name in args.cost_function_list]

    # run
    with util.logging.Logger(level=args.debug_level):
        save_for_all_measurements(max_box_distance_to_water_list=max_box_distance_to_water_list, min_standard_deviation_list=args.min_standard_deviation_list, min_measurements_correlation_list=args.min_measurements_correlation_list, cost_function_classes=cost_function_classes, model_names=args.model_list, eval_f=True, eval_df=args.DF)
        util.logging.info('Finished.')


if __name__ == "__main__":
    _main()
