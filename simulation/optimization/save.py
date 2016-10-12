import argparse
import re
import tempfile
import time

import simulation.optimization.cost_function
import simulation.optimization.job
import simulation.model.constants

import measurements.all.pw.data

import util.batch.universal.system
import util.logging
logger = util.logging.logger



def save(measurements_collection, model_names=None, cost_function_classes=None, eval_f=True, eval_df=False):
    for cost_function in simulation.optimization.cost_function.iterator(measurements_collection, model_names=model_names, cost_function_classes=cost_function_classes):
        if eval_f and not cost_function.f_available():
            logger.info('Saving cost function f value in {}'.format(cost_function.model.parameter_set_dir))
            cost_function.f()
        if eval_df and not cost_function.df_available():
            logger.info('Saving cost function df value in {}'.format(cost_function.model.parameter_set_dir))
            cost_function.df()



def save_for_all_measurements(max_box_distance_to_water=float('inf'), min_measurements_correlations=float('inf'), model_names=None, cost_function_classes=None, eval_f=True, eval_df=False):
    measurements_collection = measurements.all.pw.data.all_measurements(max_box_distance_to_water=max_box_distance_to_water, min_measurements_correlations=min_measurements_correlations)
    save(measurements_collection, model_names=model_names, cost_function_classes=cost_function_classes, eval_f=eval_f, eval_df=eval_df)



def save_for_all_measurements_with_jobs(max_box_distance_to_water=float('inf'), min_measurements_correlations=float('inf'), model_names=None, cost_function_classes=None, eval_f=True, eval_df=False, node_kind='clexpress'):
    measurements_collection = measurements.all.pw.data.all_measurements(max_box_distance_to_water=max_box_distance_to_water, min_measurements_correlations=min_measurements_correlations)
    nodes_setup = util.batch.universal.system.NodeSetup(memory=50, node_kind=node_kind, nodes=1, cpus=1, total_cpus_max=1, walltime=1)
    for cost_function in simulation.optimization.cost_function.iterator(measurements_collection, model_names=model_names, cost_function_classes=cost_function_classes):
        if (eval_f and not cost_function.f_available()) or (eval_df and not cost_function.df_available()):
                output_dir = tempfile.TemporaryDirectory(dir=simulation.model.constants.DATABASE_TMP_DIR, prefix='save_value_cost_function_tmp_').name
                ints_str = ','.join(re.findall('\d+', cost_function.model.parameter_set_dir)[-3:])
                job_name = '{cost_function_name}:{model_name}:{ints_str}'.format(cost_function_name=cost_function.name, model_name=cost_function.model.model_options.model_name, ints_str=ints_str)
                job_options = {'name': job_name, 'node_setup': nodes_setup}
                with simulation.optimization.job.CostFunctionJob(output_dir, cost_function.name, cost_function.model.model_options, max_box_distance_to_water=max_box_distance_to_water, min_measurements_correlations=min_measurements_correlations, eval_f=eval_f, eval_df=eval_df, job_options=job_options) as cost_function_job:
                    cost_function_job.start()
                time.sleep(10)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculating cost function values.')
    parser.add_argument('-m', '--max_box_distance_to_water', type=int, default=float('inf'), help='The maximal distance to water boxes to accept measurements.')
    parser.add_argument('-c', '--min_measurements_correlations', type=int, default=float('inf'), help='The minimal number of measurements used to calculate correlations.')
    parser.add_argument('-j', '--as_jobs', action='store_true', help='Run as jobs.')
    parser.add_argument('-n', '--node_kind', default='clexpress', help='Node kind to use for the jobs.')
    parser.add_argument('--DF', action='store_true', help='Eval (also) DF.')
    parser.add_argument('-d', '--debug_level', choices=util.logging.LEVELS, default='INFO', help='Print debug infos low to passed level.')
    args = parser.parse_args()
    
    with util.logging.Logger(level=args.debug_level):
        if args.as_jobs:
            save_for_all_measurements_with_jobs(max_box_distance_to_water=args.max_box_distance_to_water, min_measurements_correlations=args.min_measurements_correlations, eval_f=True, eval_df=args.DF, node_kind=args.node_kind)
        else:
            save_for_all_measurements(max_box_distance_to_water=args.max_box_distance_to_water, min_measurements_correlations=args.min_measurements_correlations, eval_f=True, eval_df=args.DF)
