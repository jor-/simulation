import os.path

import numpy as np

import ndop.optimization.constants

import util.io.fs
import util.rzcluster.job
import util.rzcluster.interact

import util.logging
logger = util.logging.logger



class CostFunctionJob(util.rzcluster.job.Job):

    def __init__(self, output_dir, parameters, cf_kind, eval_f=True, eval_df=True, write_output_file=True, **cf_kargs):
        from ndop.optimization.constants import COST_FUNCTION_NODES_SETUP_JOB
        
        super().__init__(output_dir)
        
        ## save CF options
        self.options['/cf/kind'] = cf_kind
        self.options['/cf/parameters'] = parameters
        self.options['/cf/kargs'] = cf_kargs
        
        python_script_file = os.path.join(output_dir, 'run.py')
        self.options['/cf/run_file'] = python_script_file
        
        
        ## prepare job options and init job file
        data_kind = cf_kargs['data_kind']
        try:
            job_name = cf_kargs['job_setup']['name']
        except KeyError:
            job_name = 'CF_{}_{}'.format(data_kind, cf_kind)
            if cf_kind == 'GLS':
                job_name = job_name + '_{}_{}'.format(cf_kargs['correlation_min_values'], cf_kargs['correlation_max_year_diff'])
        
        node_numbers = 1
        cpu_numbers = 1
        
        if data_kind == 'WOA':
            memory_gb = 2
            walltime_hours = 1        
        if data_kind == 'WOD':
            if cf_kind == 'GLS':
                if cf_kargs['correlation_min_values'] >= 35:
                    memory_gb = 26
                # elif cf_kargs['correlation_min_values'] >= 30:
                #     memory_gb = 26
                else:
                    memory_gb = 31
                # nodes_setup = ('f_ocean', 1, 8)
                # cpu_numbers = 8
            else:
                memory_gb = 24
        # nodes_setup = (cpu_kind, node_numbers, cpu_numbers)
        # node_kind=('westmere', 'shanghai', 'f_ocean')
        # nodes_setup = util.rzcluster.interact.NodeSetup(memory=memory_gb, node_kind=ndop.optimization.constants.COST_FUNCTION_JOB_NODE_KIND, nodes_max=node_numbers, total_cpus_max=cpu_numbers)
        nodes_setup = COST_FUNCTION_NODES_SETUP_JOB.copy()
        nodes_setup['memory'] = memory_gb
        # nodes_setup.wait_for_needed_resources()
        # nodes_setup['cpus'] = min(nodes_setup['cpus'], cpu_numbers)
        # nodes_setup = nodes_setup.tuple
            
        # queue = 'medium'
        queue = None
        walltime_hours = None
        # super().init_job_file(job_name, memory_gb, nodes_setup, queue=queue, walltime_hours=walltime_hours, write_output_file=write_output_file)
        super().init_job_file(job_name, nodes_setup, queue=queue, walltime_hours=walltime_hours, write_output_file=write_output_file)
        
        ## convert inf to negative for script
        if 'correlation_max_year_diff' in cf_kargs and cf_kargs['correlation_max_year_diff'] == float('inf'):
            cf_kargs['correlation_max_year_diff'] = -1
        
        ## write python script
        from ndop.model.constants import MODEL_PARAMETERS_FORMAT_STRING
        
        commands = ['import ndop.optimization.cost_function']
        commands += ['import util.logging']
        
        commands += ['with util.logging.Logger():']
        
        commands += ["  cf = ndop.optimization.cost_function.{}(**{})".format(cf_kind, cf_kargs)]
        
        parameters_str = str(tuple(map(lambda f: MODEL_PARAMETERS_FORMAT_STRING.format(f), parameters)))
        parameters_str = parameters_str.replace("'", '')

        if eval_f:
            commands += ['  cf.f({})'.format(parameters_str)]
        if eval_df:
            commands += ['  cf.df({})'.format(parameters_str)]
        
        
        script_str = "\n".join(commands)
        f = open(python_script_file, mode='w')
        f.write(script_str)
        util.io.fs.flush_and_close(f)

        
        ## prepare run command and write job file
        super().write_job_file('python3 {}'.format(python_script_file))

