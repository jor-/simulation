import os.path

import numpy as np

import ndop.optimization.constants

import util.batch.universal.system
import util.io.fs

import util.logging
logger = util.logging.logger



class CostFunctionJob(util.batch.universal.system.Job):

    def __init__(self, output_dir, parameters, cf_kind, eval_f=True, eval_df=True, **cf_kargs):
        from ndop.optimization.constants import COST_FUNCTION_NODES_SETUP_JOB

        super().__init__(output_dir)
        
        ## prepare job name
        data_kind = cf_kargs['data_kind']
        try:
            job_name = cf_kargs['job_setup']['name']
        except (KeyError, TypeError):
            job_name = '{}_{}'.format(data_kind, cf_kind)
            if cf_kind == 'GLS':
                job_name = job_name + '_{}_{}'.format(cf_kargs['correlation_min_values'], cf_kargs['correlation_max_year_diff'])

        ## prepare job_setup
        if 'job_setup' in cf_kargs:
            job_setup = cf_kargs['job_setup']
            del cf_kargs['job_setup']
        else:
            job_setup = None

        ## save CF options
        self.options['/cf/kind'] = cf_kind
        self.options['/cf/parameters'] = parameters
        for key, value in cf_kargs.items():
            if value is not None:
                self.options['/cf/{}'.format(key)] = value

        python_script_file = os.path.join(output_dir, 'run.py')
        self.options['/cf/run_file'] = python_script_file

        ## prepare job options and init job file
        node_numbers = 1
        cpu_numbers = 1

        if data_kind == 'WOA':
            memory_gb = 2
            walltime_hours = 1
        if data_kind.startswith('WOD'):
            if cf_kind == 'GLS':
                if cf_kargs['correlation_min_values'] >= 35:
                    memory_gb = 26
                elif cf_kargs['correlation_min_values'] >= 30:
                    memory_gb = 31
                else:
                    memory_gb = 40
            else:
                memory_gb = 24
        nodes_setup = COST_FUNCTION_NODES_SETUP_JOB.copy()
        nodes_setup['memory'] = memory_gb
        queue = None
        super().init_job_file(job_name, nodes_setup, queue=queue)

        ## convert inf to negative for script
        if 'correlation_max_year_diff' in cf_kargs and cf_kargs['correlation_max_year_diff'] == float('inf'):
            cf_kargs['correlation_max_year_diff'] = -1

        ## write python script
        commands = ['import util.logging']
        commands += ['with util.logging.Logger():']
        commands += ['    import ndop.optimization.cost_function']
        commands += ["    cf_kargs = {}".format(cf_kargs)]
        if job_setup is not None:
            commands += ['    import util.batch.universal.system']
            commands += ["    job_setup = {}"]
            for setup_name in ('spinup', 'derivative', 'trajectory'):
                if setup_name in job_setup:
                    nodes_setup = job_setup[setup_name]['nodes_setup']
                    nodes_setup_str = "util.batch.universal.system.NodeSetup(memory={}, node_kind='{}', nodes={}, cpus={}, walltime={})".format(nodes_setup.memory, nodes_setup.node_kind, nodes_setup.nodes, nodes_setup.cpus, nodes_setup.walltime)
                    job_setup_str = "{'" + setup_name + "':{'nodes_setup':" + nodes_setup_str + "}}"
                    commands += ["    job_setup.update({})".format(job_setup_str)]
            commands += ["    cf_kargs.update({'job_setup':job_setup})"]
        commands += ['    cf = ndop.optimization.cost_function.{}(**cf_kargs)'.format(cf_kind)]

        from ndop.model.constants import MODEL_PARAMETERS_FORMAT_STRING
        parameters_str = str(tuple(map(lambda f: MODEL_PARAMETERS_FORMAT_STRING.format(f), parameters)))
        parameters_str = parameters_str.replace("'", '')
        if eval_f:
            commands += ['    cf.f({})'.format(parameters_str)]
        if eval_df:
            commands += ['    cf.df({})'.format(parameters_str)]


        script_str = "\n".join(commands)
        f = open(python_script_file, mode='w')
        f.write(script_str)
        util.io.fs.flush_and_close(f)

        ## prepare run command and write job file
        super().write_job_file('python3 {}'.format(python_script_file))

