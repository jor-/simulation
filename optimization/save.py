import argparse
import os
import tempfile
import time

import numpy as np

import ndop.optimization.cost_function
import ndop.optimization.constants
import ndop.optimization.job
import ndop.model.eval

import util.logging
logger = util.logging.logger

from ndop.constants import BASE_DIR



def save(parameter_sets=range(9999), data_kind='WOA', eval_f=True, eval_df=True, as_jobs=False):
    from ndop.model.constants import MODEL_OUTPUT_DIR, MODEL_TIME_STEP_DIRNAME, MODEL_PARAMETERS_SET_DIRNAME, MODEL_PARAMETERS_FILENAME, MODEL_SPINUP_DIRNAME, MODEL_DERIVATIVE_DIRNAME

    for parameter_set_number in parameter_sets:

        ## get cf family
        cost_function_family = None

        time_step = 1
        time_step_dirname = MODEL_TIME_STEP_DIRNAME.format(time_step)
        time_step_dir = os.path.join(MODEL_OUTPUT_DIR, time_step_dirname)

        parameter_set_dirname = MODEL_PARAMETERS_SET_DIRNAME.format(parameter_set_number)
        parameter_set_dir = os.path.join(time_step_dir, parameter_set_dirname)
        parameters_file = os.path.join(parameter_set_dir, MODEL_PARAMETERS_FILENAME)

        if os.path.exists(parameters_file):
            model = ndop.model.eval.Model()

            spinup_dir = os.path.join(parameter_set_dir, MODEL_SPINUP_DIRNAME)
            last_run_dir = model.get_last_run_dir(spinup_dir)

            if last_run_dir is not None:
                years = model.get_total_years(last_run_dir)
                tolerance = model.get_real_tolerance(last_run_dir)
                time_step = model.get_time_step(last_run_dir)

                ## create cost functions
                cf_kargs = {'data_kind':data_kind, 'spinup_options':{'years':years, 'tolerance':tolerance, 'combination':'and'}, 'job_setup':{'name': 'SCF_' + data_kind}}
                cost_function_family = ndop.optimization.cost_function.Family(**cf_kargs)

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
                        derivative_dir = os.path.join(parameter_set_dir, MODEL_DERIVATIVE_DIRNAME)
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
                            with ndop.optimization.job.CostFunctionJob(output_dir, p, cf.kind, eval_f=eval_f, eval_df=eval_df, **cf_kargs) as cf_job:
                                cf_job.start()
                            time.sleep(10)

            except ValueError as e:
                logger.exception(e)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculating cost function values.')

    parser.add_argument('-k', '--kind_of_cost_function', choices=tuple(ndop.optimization.cost_function.Family.member_classes.keys()), help='The kind of the cost function to chose.')
    parser.add_argument('-f', '--first', type=int, default=0, help='First parameter set number for which to calculate the values.')
    parser.add_argument('-l', '--last', type=int, default=9999, help='Last parameter set number for which to calculate the values.')
    parser.add_argument('-d', '--debug', action='store_true', help='Print debug infos.')
    parser.add_argument('-j', '--as_jobs', action='store_true', help='Run as jobs.')
    parser.add_argument('--DF', action='store_true', help='Eval (also) DF.')
    parser.add_argument('--version', action='version', version='%(prog)s 0.1')

    args = parser.parse_args()
    parameter_sets = range(args.first, args.last+1)
    
    with util.logging.Logger(disp_stdout=args.debug):
        save(parameter_sets=parameter_sets, data_kind=args.kind_of_cost_function, eval_df=args.DF, as_jobs=args.as_jobs)