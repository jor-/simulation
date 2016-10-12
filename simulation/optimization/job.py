import os

import numpy as np

import simulation.constants
import simulation.model.constants
import simulation.model.options
import simulation.optimization.constants

import measurements.constants

import util.batch.universal.system
import util.io.fs
import util.io.env

import util.logging
logger = util.logging.logger



class CostFunctionJob(util.batch.universal.system.Job):

    def __init__(self, output_dir, cf_kind, model_options, model_job_options=None, max_box_distance_to_water=float('inf'), min_measurements_correlations=float('inf'), eval_f=True, eval_df=True, job_options=None):
        from simulation.optimization.constants import COST_FUNCTION_NODES_SETUP_JOB
        
        logger.debug('Initiating cost function job with cf_kind {}, eval_f {} and eval_df {}.'.format(cf_kind, eval_f, eval_df))
        
        model_options = simulation.model.options.as_model_options(model_options)

        super().__init__(output_dir)
        
        ## save CF options
        self.options['/cf/kind'] = cf_kind
        self.options['/cf/model_options'] = repr(model_options)
        self.options['/cf/model_job_options'] = repr(model_job_options)
        self.options['/cf/max_box_distance_to_water'] = max_box_distance_to_water
        self.options['/cf/min_measurements_correlations'] = min_measurements_correlations
        
        ## prepare job options
        if job_options is None:
            job_options = {}
        
        ## prepare job name
        try:
            job_name = job_options['name']
        except KeyError:
            job_name = cf_kind
            if cf_kind == 'GLS':
                job_name = job_name + '_{min_measurements_correlations}'.format(min_measurements_correlations=min_measurements_correlations)
            job_name = job_name + '_' + model_options.model_name
            if max_box_distance_to_water is not None and max_box_distance_to_water != float('inf'):
                job_name = job_name + '_N{max_box_distance_to_water:d}'.format(max_box_distance_to_water=max_box_distance_to_water)

        ## prepare node setup
        try:
            nodes_setup = job_options['nodes_setup']
        except KeyError:
            nodes_setup = COST_FUNCTION_NODES_SETUP_JOB.copy()
            if eval_df:
                nodes_setup['memory'] = nodes_setup['memory'] + 5
            if cf_kind == 'GLS':
                nodes_setup['memory'] = nodes_setup['memory'] + 20
        
        ## init job file
        queue = None
        super().init_job_file(job_name, nodes_setup, queue=queue)
        
        ## write python script
        commands = ['import numpy as np']
        commands += ['import simulation.model.options']
        commands += ['import simulation.optimization.cost_function']
        commands += ['import measurements.all.pw.data']
        commands += ['import util.batch.universal.system']
        commands += ['import util.logging']

        if max_box_distance_to_water == float('inf'):
            max_box_distance_to_water = None
        if min_measurements_correlations == float('inf'):
            min_measurements_correlations = None
        
        commands += ['with util.logging.Logger():']
        commands += ['    model_options = {model_options!r}'.format(model_options=model_options)]
        commands += ['    measurements_collection = measurements.all.pw.data.all_measurements(max_box_distance_to_water={max_box_distance_to_water}, min_measurements_correlations={min_measurements_correlations}, tracers=model_options.tracers)'.format(max_box_distance_to_water=max_box_distance_to_water, min_measurements_correlations=min_measurements_correlations)]
        
        if model_job_options is not None:
            commands += ['    job_options = {model_job_options!r}'.format(model_job_options=model_job_options)]
        else:
            commands += ['    job_options = None']
        commands += ['    cf = simulation.optimization.cost_function.{cf_kind}(measurements_collection=measurements_collection, model_options=model_options, job_options=job_options)'.format(cf_kind=cf_kind)]
        
        parameters_str = ','.join(map(lambda f: simulation.model.constants.DATABASE_PARAMETERS_FORMAT_STRING.format(f), model_options.parameters))
        commands += ['    cf.parameters = ({})'.format(parameters_str)]
        if eval_f:
            commands += ['    cf.f()']
        if eval_df:
            commands += ['    cf.df()']
        commands += ['']

        script_str = os.linesep.join(commands)
        script_str = script_str.replace('array', 'np.array')

        python_script_file = os.path.join(output_dir, 'run.py')
        with open(python_script_file, mode='w') as f:
            f.write(script_str)
            f.flush()

        ## prepare run command and write job file
        def export_env_command(env_name):
            try:
                env_value = util.io.env.load(env_name)
            except util.io.env.EnvironmentLookupError:
                return ''
            else:
                return 'export {env_name}={env_value}'.format(env_name=env_name, env_value=env_value)
        env_names = [simulation.constants.BASE_DIR_ENV_NAME, simulation.constants.SIMULATION_OUTPUT_DIR_ENV_NAME, simulation.constants.METOS3D_DIR_ENV_NAME, measurements.constants.BASE_DIR_ENV_NAME, util.batch.universal.system.BATCH_SYSTEM_ENV_NAME, util.io.env.PYTHONPATH_ENV_NAME]
        env_commands = [export_env_command(env_name) for env_name in env_names]
        env_commands = [env_command for env_command in env_commands if len(env_command) > 0]
        export_env_command = os.linesep.join(env_commands)
            
        python_command = util.batch.universal.system.BATCH_SYSTEM.commands['python']
        run_command = '{python_command} {python_script_file}'.format(python_command=python_command, python_script_file=python_script_file)
        
        super().write_job_file(run_command, pre_run_command=export_env_command, modules=['intel16'])

