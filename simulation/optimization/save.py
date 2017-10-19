import simulation
import simulation.optimization.cost_function
import simulation.optimization.job
import simulation.model.constants
import simulation.model.options

import measurements.all.data

import util.batch.universal.system
import util.logging


def save(cost_functions, model_names=None, eval_f=True, eval_df=False):
    for cost_function in simulation.optimization.cost_function.iterator(cost_functions, model_names=model_names):
        eval_this_f = eval_f and not cost_function.f_available()
        eval_this_df = eval_df and not cost_function.df_available()
        if eval_this_f or eval_this_df:
            if eval_this_f:
                try:
                    cost_function.f()
                except Exception:
                    util.logging.error('Model function could not be evaluated.', exc_info=True)
                else:
                    util.logging.info('Saving cost function {cost_function_name} f value in {model_parameter_dir}.'.format(
                        cost_function_name=cost_function,
                        model_parameter_dir=cost_function.model.parameter_set_dir))
            if eval_this_df:
                try:
                    cost_function.df()
                except Exception:
                    util.logging.error('Model function derivative could not be evaluated.', exc_info=True)
                else:
                    util.logging.info('Saving cost function {cost_function_name} df value in {model_parameter_dir}.'.format(
                        cost_function_name=cost_function,
                        model_parameter_dir=cost_function.model.parameter_set_dir))
        else:
            util.logging.debug('Cost function values {cost_function_name} in {model_parameter_dir} with eval_f={eval_f} and eval_df={eval_df} are already available.'.format(
                cost_function_name=cost_function.name,
                model_parameter_dir=cost_function.model.parameter_set_dir,
                eval_f=eval_f,
                eval_df=eval_df))


def save_for_all_measurements_serial(cost_function_names=None, model_names=None, min_standard_deviations=None, min_measurements_correlations=None, max_box_distance_to_water=None, eval_f=True, eval_df=False):
    # get cost_function_classes
    if cost_function_names is None:
        cost_function_classes = None
    else:
        cost_function_classes = [getattr(simulation.optimization.cost_function, cost_function_name) for cost_function_name in cost_function_names]
    # init cost functions
    model_options = simulation.model.options.ModelOptions()
    model_options.spinup_options = {'years': 1, 'tolerance': 0.0, 'combination': 'or'}
    cost_functions = simulation.optimization.cost_function.cost_functions_for_all_measurements(
        min_standard_deviations=min_standard_deviations,
        min_measurements_correlations=min_measurements_correlations,
        max_box_distance_to_water=max_box_distance_to_water,
        cost_function_classes=cost_function_classes,
        model_options=model_options)
    if eval_df:
        for cost_function in cost_functions:
            cost_function.include_initial_concentrations_factor_by_default = True
    # save values
    save(cost_functions, model_names=model_names, eval_f=eval_f, eval_df=eval_df)


def save_for_all_measurements_as_jobs(cost_function_names=None, model_names=None, min_standard_deviations=None, min_measurements_correlations=None, max_box_distance_to_water=None, eval_f=True, eval_df=False, node_kind=None):
    if eval_f or eval_df:
        model_job_options = None
        include_initial_concentrations_factor_by_default = True

        nodes_setup = simulation.optimization.constants.COST_FUNCTION_NODES_SETUP_JOB.copy()
        if node_kind is not None:
            nodes_setup.node_kind = node_kind
        cost_function_job_options = {'nodes_setup': nodes_setup}

        if cost_function_names is None:
            cost_function_names = simulation.optimization.cost_function.ALL_COST_FUNCTION_NAMES

        for cost_function_name in cost_function_names:
            cost_function_class = getattr(simulation.optimization.cost_function, cost_function_name)

            for model_name in model_names:
                model = simulation.model.cache.Model(job_options=model_job_options)

                for model_options in model.iterator(model_names=[model_name]):
                    measurements_object = measurements.all.data.all_measurements(
                        tracers=model_options.tracers,
                        min_standard_deviations=min_standard_deviations,
                        min_measurements_correlations=min_measurements_correlations,
                        max_box_distance_to_water=max_box_distance_to_water)
                    cost_function = cost_function_class(
                        measurements_object,
                        model_options=model_options,
                        model_job_options=model_job_options,
                        include_initial_concentrations_factor_by_default=include_initial_concentrations_factor_by_default)

                    if (eval_f and not cost_function.f_available()) or (eval_df and not cost_function.df_available()):
                        with simulation.optimization.job.CostFunctionJob(
                                cost_function_name,
                                model_options,
                                model_job_options=model_job_options,
                                min_standard_deviations=min_standard_deviations,
                                min_measurements_correlations=min_measurements_correlations,
                                max_box_distance_to_water=max_box_distance_to_water,
                                eval_f=eval_f,
                                eval_df=eval_df,
                                cost_function_job_options=cost_function_job_options,
                                include_initial_concentrations_factor_by_default=include_initial_concentrations_factor_by_default) as cf_job:
                            cf_job.start()

                        util.logging.info('Starting cost function job {cost_function_name} for values in {model_parameter_dir} with eval_f={eval_f} and eval_df={eval_df}.'.format(
                            cost_function_name=cost_function_name,
                            model_parameter_dir=model.parameter_set_dir,
                            eval_f=eval_f,
                            eval_df=eval_df))
                    else:
                        util.logging.debug('Cost function values {cost_function_name} in {model_parameter_dir} with eval_f={eval_f} and eval_df={eval_df} are already available.'.format(
                            cost_function_name=cost_function_name,
                            model_parameter_dir=model.parameter_set_dir,
                            eval_f=eval_f,
                            eval_df=eval_df))


def save_for_all_measurements(cost_function_names=None, model_names=None, min_standard_deviations=None, min_measurements_correlations=None, max_box_distance_to_water=None, eval_f=True, eval_df=False, as_jobs=False, node_kind=None):
    if as_jobs:
        save_for_all_measurements_as_jobs(
            cost_function_names=cost_function_names,
            model_names=model_names,
            min_standard_deviations=min_standard_deviations,
            min_measurements_correlations=min_measurements_correlations,
            max_box_distance_to_water=max_box_distance_to_water,
            eval_f=eval_f,
            eval_df=eval_df,
            node_kind=node_kind)
    else:
        save_for_all_measurements_serial(
            cost_function_names=cost_function_names,
            model_names=model_names,
            min_standard_deviations=min_standard_deviations,
            min_measurements_correlations=min_measurements_correlations,
            max_box_distance_to_water=max_box_distance_to_water,
            eval_f=eval_f,
            eval_df=eval_df)


# *** main function for script call *** #

def _main():

    # parse arguments
    import argparse

    parser = argparse.ArgumentParser(description='Calculating cost function values.')
    parser.add_argument('--min_standard_deviations', nargs='+', type=float, default=None, help='The minimal standard deviations assumed for the measurement errors applied for each dataset.')
    parser.add_argument('--min_measurements_correlations', nargs='+', type=int, default=float('inf'), help='The minimal number of measurements used to calculate correlations applied to each dataset.')
    parser.add_argument('--max_box_distance_to_water', type=int, default=None, help='The maximal distances to water boxes to accept measurements.')
    parser.add_argument('--cost_functions', type=str, default=None, nargs='+', help='The cost functions to evaluate.')
    parser.add_argument('--model_names', type=str, default=None, choices=simulation.model.constants.MODEL_NAMES, nargs='+', help='The models to evaluate.')
    parser.add_argument('--DF', action='store_true', help='Eval (also) DF.')
    parser.add_argument('--as_jobs', action='store_true', help='Eval as batch jobs.')
    parser.add_argument('--node_kind', default=None, help='The kind of nodes to use for the batch jobs.')
    parser.add_argument('--debug_level', choices=util.logging.LEVELS, default='INFO', help='Print debug infos low to passed level.')
    parser.add_argument('--version', action='version', version='%(prog)s {}'.format(simulation.__version__))
    args = parser.parse_args()

    # run
    with util.logging.Logger(level=args.debug_level):
        save_for_all_measurements(
            cost_function_names=args.cost_functions,
            model_names=args.model_names,
            min_standard_deviations=args.min_standard_deviations,
            min_measurements_correlations=args.min_measurements_correlations,
            max_box_distance_to_water=args.max_box_distance_to_water,
            eval_f=True,
            eval_df=args.DF,
            node_kind=args.node_kind,
            as_jobs=args.as_jobs)
        util.logging.info('Finished.')


if __name__ == "__main__":
    _main()
