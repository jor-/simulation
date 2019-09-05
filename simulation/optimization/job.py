import os
import tempfile

import simulation.constants
import simulation.model.constants
import simulation.model.options
import simulation.optimization.constants

import measurements.constants

import util.batch.universal.system
import util.io.env

import util.logging


class CostFunctionJob(util.batch.universal.system.Job):

    def __init__(self, cf_kind, model_options,
                 output_dir=None, model_job_options=None,
                 min_measurements_standard_deviations=None, min_measurements_correlations=None,
                 min_standard_deviations=None, correlation_decomposition_min_value_D=None,
                 max_box_distance_to_water=None, eval_f=True, eval_df=False, eval_d2f=False,
                 cost_function_job_options=None, include_initial_concentrations_factor_to_model_parameters=False,
                 remove_output_dir_on_close=False):
        util.logging.debug('Initiating cost function job with cf_kind {}, eval_f {} and eval_df {} and eval_d2f {}.'.format(cf_kind, eval_f, eval_df, eval_d2f))

        # if no output dir, use tmp output dir
        if output_dir is None:
            output_dir = simulation.model.constants.DATABASE_TMP_DIR
            os.makedirs(output_dir, exist_ok=True)
            output_dir = tempfile.mkdtemp(dir=output_dir, prefix='cost_function_tmp_')

        # init job object
        super().__init__(output_dir, remove_output_dir_on_close=remove_output_dir_on_close)

        # convert model options
        model_options = simulation.model.options.as_model_options(model_options)

        # save CF options
        self.options['/cf/kind'] = cf_kind
        self.options['/cf/model_options'] = repr(model_options)
        self.options['/cf/model_job_options'] = repr(model_job_options)
        self.options['/cf/max_box_distance_to_water'] = max_box_distance_to_water
        self.options['/cf/min_measurements_standard_deviations'] = min_measurements_standard_deviations
        self.options['/cf/min_measurements_correlations'] = min_measurements_correlations
        self.options['/cf/min_standard_deviations'] = min_standard_deviations
        self.options['/cf/correlation_decomposition_min_value_D'] = correlation_decomposition_min_value_D
        self.options['/cf/include_initial_concentrations_factor_to_model_parameters'] = include_initial_concentrations_factor_to_model_parameters

        # prepare job options
        if cost_function_job_options is None:
            cost_function_job_options = {}

        # prepare job name
        try:
            job_name = cost_function_job_options['name']
        except KeyError:
            job_name = cf_kind
            if cf_kind in ['WLS', 'GLS']:
                job_name = job_name + f'_{min_measurements_standard_deviations}'
            if cf_kind == 'GLS':
                job_name = job_name + f'_{min_measurements_correlations}'
            job_name = job_name + '_' + model_options.model_name + '_' + str(model_options.time_step)
            if max_box_distance_to_water is not None and max_box_distance_to_water != float('inf'):
                job_name = job_name + f'_N{max_box_distance_to_water:d}'

        # get node setup
        batch_system = util.batch.universal.system.BATCH_SYSTEM
        try:
            nodes_setup = cost_function_job_options['nodes_setup']
        except KeyError:
            nodes_setup = simulation.optimization.constants.NODES_SETUP_JOB.copy()

        # set walltime if not set
        if nodes_setup.walltime is None:
            walltime = model_options.tracers_len
            if eval_df:
                walltime += model_options.tracers_len * model_options.parameters_len * 2
            if eval_d2f:
                walltime += model_options.tracers_len * model_options.parameters_len * (model_options.parameters_len + 1) * 0.5
                if not eval_df:
                    walltime += model_options.tracers_len * model_options.parameters_len
            try:
                node_kind = nodes_setup['node_kind']
            except KeyError:
                pass
            else:
                try:
                    max_walltime = batch_system.max_walltime[node_kind]
                except KeyError:
                    pass
                else:
                    walltime = min(walltime, max_walltime)
            nodes_setup.walltime = walltime

        # set memory if not set
        try:
            nodes_setup.memory
        except AttributeError:
            memory = 30
            if eval_df:
                memory += 5
            if eval_d2f:
                memory += 5
            if cf_kind == 'GLS':
                memory += 20
            nodes_setup.memory = memory

        # init job file
        queue = None
        self.set_job_options(job_name, nodes_setup, queue=queue)

        # write python script
        if max_box_distance_to_water == float('inf'):
            max_box_distance_to_water = None
        if min_measurements_correlations == float('inf'):
            min_measurements_correlations = None

        commands = []
        commands += ['import util.logging']
        commands += ['with util.logging.Logger():']
        commands += ['    import numpy as np']
        commands += ['    import util.batch.universal.system']
        commands += ['    import measurements.all.data']
        commands += ['    import simulation.model.options']
        commands += ['    import simulation.optimization.cost_function']

        commands += [f'    model_options = {model_options!r}']
        commands += [f'    measurements_object = measurements.all.data.all_measurements(tracers=model_options.tracers, min_measurements_standard_deviation={min_measurements_standard_deviations}, min_measurements_correlation={min_measurements_correlations}, min_standard_deviation={min_standard_deviations}, correlation_decomposition_min_value_D={correlation_decomposition_min_value_D}, max_box_distance_to_water={max_box_distance_to_water}, water_lsm="TMM", sample_lsm="TMM")']

        if model_job_options is not None:
            commands += [f'    model_job_options = {model_job_options!r}']
        else:
            commands += ['    model_job_options = None']
        commands += [f'    cf = simulation.optimization.cost_function.{cf_kind}(measurements_object=measurements_object, model_options=model_options, model_job_options=model_job_options, include_initial_concentrations_factor_to_model_parameters={include_initial_concentrations_factor_to_model_parameters})']

        if eval_f:
            commands += ['    cf.f()']
        if eval_df:
            commands += ['    cf.df(derivative_order=1)']
        if eval_d2f:
            commands += ['    cf.df(derivative_order=2)']
        commands += ['']

        script_str = os.linesep.join(commands)
        script_str = script_str.replace('array', 'np.array')

        python_script_file = os.path.join(output_dir, 'run.py')
        with open(python_script_file, mode='w') as f:
            f.write(script_str)
            f.flush()

        # prepare run command and write job file
        def export_env_command(env_name):
            try:
                env_value = util.io.env.load(env_name)
            except util.io.env.EnvironmentLookupError:
                return ''
            else:
                return 'export {env_name}={env_value}'.format(env_name=env_name, env_value=env_value)
        env_names = [simulation.constants.BASE_DIR_ENV_NAME, simulation.constants.SIMULATION_OUTPUT_DIR_ENV_NAME, simulation.constants.METOS3D_DIR_ENV_NAME, measurements.constants.BASE_DIR_ENV_NAME, util.batch.universal.system.BATCH_SYSTEM_ENV_NAME, util.io.env.PYTHONPATH_ENV_NAME]
        pre_commands = [export_env_command(env_name) for env_name in env_names]
        pre_commands.append(batch_system.pre_command('python'))

        pre_commands = [pre_command for pre_command in pre_commands if len(pre_command) > 0]
        pre_command = os.linesep.join(pre_commands)

        python_command = batch_system.command('python')
        command = '{python_command} {python_script_file}'.format(python_command=python_command, python_script_file=python_script_file)

        super().write_job_file(command, pre_command=pre_command, use_mpi=False, use_conda=True, add_timing=True)
