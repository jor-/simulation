import argparse
import os
import tempfile
import time

import numpy as np

import simulation.optimization.cost_function
import simulation.optimization.constants
import simulation.optimization.job
import simulation.model.eval

import util.batch.universal.system
import util.logging
logger = util.logging.logger


def save(model_name='dop_po4', time_step=1, parameter_sets=range(9999), data_kind='WOA', eval_f=True, eval_df=True, as_jobs=False, node_kind='clexpress'):
    from simulation.model.constants import DATABASE_OUTPUT_DIR, DATABASE_MODEL_DIRNAME, DATABASE_TIME_STEP_DIRNAME, DATABASE_PARAMETERS_DIRNAME, DATABASE_PARAMETERS_FILENAME, DATABASE_SPINUP_DIRNAME, DATABASE_DERIVATIVE_DIRNAME
    
    ## get time step dir
    model_dirname = DATABASE_MODEL_DIRNAME.format(model_name)
    model_dir = os.path.join(DATABASE_OUTPUT_DIR, model_dirname)
    time_step_dirname = DATABASE_TIME_STEP_DIRNAME.format(time_step)
    time_step_dir = os.path.join(model_dir, time_step_dirname)
    
    ## create model
    model = simulation.model.eval.Model()

    ## save for all parameter sets
    for parameter_set_number in parameter_sets:

        ## get parameter
        cost_function_family = None

        parameter_set_dirname = DATABASE_PARAMETERS_DIRNAME.format(parameter_set_number)
        parameter_set_dir = os.path.join(time_step_dir, parameter_set_dirname)
        parameters_file = os.path.join(parameter_set_dir, DATABASE_PARAMETERS_FILENAME)

        ## create cost functions
        if os.path.exists(parameters_file):
            spinup_dir = os.path.join(parameter_set_dir, DATABASE_SPINUP_DIRNAME)
            last_run_dir = model.last_run_dir(spinup_dir)

            if last_run_dir is not None:
                cf_kargs = {'data_kind': data_kind, 'model_options': {'spinup_options': {'years':1,}, 'time_step': time_step, 'total_concentration_factor_included_in_parameters': True}, 'job_setup':{'name': 'SCF_' + data_kind}}
                cost_function_family = simulation.optimization.cost_function.Family(**cf_kargs)
                
        ## eval cf family
        if cost_function_family is not None:
            p = np.loadtxt(parameters_file)
            try:
                ## eval cf by itself
                if not as_jobs:
                    ## eval f
                    if eval_f:
                        cost_function_family.f(p)
                        cost_function_family.f_normalized(p)

                    ## eval df
                    if eval_df:
                        derivative_dir = os.path.join(parameter_set_dir, DATABASE_DERIVATIVE_DIRNAME)
                        if os.path.exists(derivative_dir):
                            cost_function_family.df(p)
                    
                ## eval cf as job
                else:
                    for cf in cost_function_family.family:
                        if (eval_f and not cf.f_available(p)) or (eval_df and not cf.df_available(p)):
                            from util.constants import TMP_DIR
                            output_dir = tempfile.TemporaryDirectory(dir=TMP_DIR, prefix='save_value_cost_function_tmp_').name
                            cf_kargs = cf.kargs
                            cf_kargs['job_setup'] = {'name': '{}:{}'.format(cf, parameter_set_number)}
                            nodes_setup = util.batch.universal.system.NodeSetup(memory=50, node_kind=node_kind, nodes=1, cpus=1, total_cpus_max=1, walltime=1)
                            with simulation.optimization.job.CostFunctionJob(output_dir, p, cf.kind, eval_f=eval_f, eval_df=eval_df, nodes_setup=nodes_setup, **cf_kargs) as cf_job:
                                cf_job.start()
                            time.sleep(10)

            except ValueError as e:
                logger.exception(e)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculating cost function values.')

    parser.add_argument('-k', '--kind_of_cost_function', required=True, choices=tuple(simulation.optimization.cost_function.Family.member_classes.keys()), help='The kind of the cost function to chose.')
    parser.add_argument('-f', '--first', type=int, default=0, help='First parameter set number for which to calculate the values.')
    parser.add_argument('-l', '--last', type=int, default=9999, help='Last parameter set number for which to calculate the values.')
    parser.add_argument('-d', '--debug', action='store_true', help='Print debug infos.')
    parser.add_argument('-j', '--as_jobs', action='store_true', help='Run as jobs.')
    parser.add_argument('-n', '--node_kind', default='clexpress', help='Node kind to use for the jobs.')
    parser.add_argument('--DF', action='store_true', help='Eval (also) DF.')
    parser.add_argument('--version', action='version', version='%(prog)s 0.1')

    args = parser.parse_args()
    parameter_sets = range(args.first, args.last+1)
    
    with util.logging.Logger(disp_stdout=args.debug):
        save(parameter_sets=parameter_sets, data_kind=args.kind_of_cost_function, eval_df=args.DF, as_jobs=args.as_jobs, node_kind=args.node_kind)
