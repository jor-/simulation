import os
import time
import subprocess
import re
import numpy as np

import logging
logger = logging.getLogger(__name__)

from util.rzcluster.job import Job
import util.io



class Metos3D_Job(Job):
    
    def __init__(self, output_path, wait_pause_seconds=30, force_load=False):
        Job.__init__(self, output_path, wait_pause_seconds, force_load)
    
    
    
    @property
    def last_spinup_line(self):
        opt = self.options
        output_file = opt['/job/output_file']
        
        self.wait_until_finished()
        
        # 9.704s 0010 Spinup Function norm 2.919666257647e+00
        last_spinup_line = None
        with open(output_file) as f:
            for line in f.readlines():
                if 'Spinup Function norm' in line:
                    last_spinup_line = line
        
        return last_spinup_line
    
    
    @property
    def last_year(self):
        # 9.704s 0010 Spinup Function norm 2.919666257647e+00
        last_spinup_line = self.last_spinup_line
        
        if last_spinup_line is not None:
            last_spinup_line = last_spinup_line.strip()
            last_spinup_year_str = last_spinup_line.split()[1]
            last_spinup_year = int(last_spinup_year_str) + 1
        else:
            last_spinup_year = 0
        
        return last_spinup_year
    
    
    @property
    def last_tolerance(self):
        # 9.704s 0010 Spinup Function norm 2.919666257647e+00
        last_spinup_line = self.last_spinup_line
        
        if last_spinup_line is not None:
            last_spinup_line = last_spinup_line.strip()
            last_spinup_tolerance_str = last_spinup_line.split()[5]
            last_spinup_tolerance = float(last_spinup_tolerance_str)
        else:
            last_spinup_tolerance = float('inf')
        
        return last_spinup_tolerance
    
    
    @property
    def time_step(self):
        from ndop.model.constants import MODEL_TIME_STEP_SIZE_MAX
        
        opt = self.options
        metos3d_opt_file = opt['/metos3d/option_file']
        with open(metos3d_opt_file) as metos3d_opt_file_object:
            metos3d_opt_lines = metos3d_opt_file_object.readlines()
        
        for metos3d_opt_line in metos3d_opt_lines:
            if re.search('Metos3DTimeStepCount', metos3d_opt_line) is not None:
                time_step_count = int(re.findall('\d+', metos3d_opt_line)[1])
                time_step = int(MODEL_TIME_STEP_SIZE_MAX / time_step_count)
        
        return time_step
    
    
    @property
    def tracer_input_path(self):
        opt = self.options
        
        try:
            input_dir = opt['/metos3d/tracer_input_path']
            input_filename = opt['/metos3d/po4_input_filename']
        except KeyError:
            input_filename = None
        
        if input_filename is not None:
            input_file = os.path.join(input_dir, input_filename)
            real_input_file = os.path.realpath(input_file)
            tracer_input_path = os.path.dirname(real_input_file)
        else:
            tracer_input_path = None
        
        return tracer_input_path
    
    
    
    def update_output_path(self, new_output_path):
        opt = self.options
        old_output_path = opt['/metos3d/output_path']
        
        if old_output_path.endswith('/'):
            old_output_path = old_output_path[:-1]
        if new_output_path.endswith('/'):
            new_output_path = new_output_path[:-1]
        
        opt.replace_all_str_options(old_output_path, new_output_path)
    
    
    
#     def init(self, model_parameters, years, tolerance, time_step=1, write_trajectory=False, tracer_input_path=None, nodes_setup=None, nodes_max=None, job_name_prefix=''):
    def init(self, model_parameters, years, tolerance, time_step=1, write_trajectory=False, tracer_input_path=None, job_setup=None):
        from ndop.model.constants import JOB_OPTIONS_FILENAME, JOB_MEMORY_GB, JOB_MIN_CPUS, MODEL_PARAMETERS_FORMAT_STRING, MODEL_TIME_STEP_SIZE_MAX, METOS_PATH_1, METOS_PATH_2
        from util.rzcluster.constants import NODES_MAX

        logger.debug('Initialising job with job_setup {}.'.format(job_setup))
        
        
        ## check input
        if MODEL_TIME_STEP_SIZE_MAX % time_step != 0:
            raise ValueError('Wrong time_step passed. ' + str(MODEL_TIME_STEP_SIZE_MAX) + ' has to be divisible by time_step. But time_step is ' + str(time_step) + '.')
        
        
        ## unpack job setup
        if job_setup is not None:
            try:
                job_name = job_setup['name']
            except KeyError:
                job_name = ''
            try:
                nodes_setup = job_setup['nodes_setup']
            except KeyError:
                nodes_setup = None
            try:
                nodes_max = job_setup['nodes_max']
            except KeyError:
                nodes_max = None
        else:
            job_name = ''
            nodes_setup = None
            nodes_max = None
        
        ## prepare job name
        if len(job_name) > 0:
            job_name += '_'
        job_name += '{}_{}'.format(years, time_step)
        
        ## chose walltime
        max_years_express = 250
        if years <= max_years_express:
            walltime_hours = 3
        else:
            walltime_hours = 240
        
        
        ## used passed nodes setup
        if nodes_setup is not None:
#             (cpu_kind, nodes, cpus) = nodes_setup
            Job.init(self, job_name, JOB_MEMORY_GB, nodes_setup, walltime_hours=walltime_hours)
        
        ## init job with best node setup, if non passed
        else:
            if years == 1:
                cpus_min = 1
                
                ## max one node
                if nodes_max is not None:
                    nodes_max = util.io.get_sequence_from_values_or_file(nodes_max)
                    nodes_max = list(nodes_max)
                    for i in range(len(nodes_max)):
                        nodes_max[i] = min(nodes_max[i], 1)
                else:
                    nodes_max = (1,) * len(NODES_MAX)
            elif years <= max_years_express:
                cpus_min = 16
            else:
                cpus_min = JOB_MIN_CPUS
                
            Job.init_best(self, job_name, JOB_MEMORY_GB, cpus_min=cpus_min, nodes_max=nodes_max, walltime_hours=walltime_hours)
            
#             (cpu_kind, nodes, cpus) = util.rzcluster.interact.wait_for_needed_resources(JOB_MEMORY_GB, cpus_min=cpus_min, nodes_max=nodes_max, node_left_free=JOB_NODES_LEFT_FREE)
#         
#         
#         ## chose queue
#         if cpu_kind in ('f_ocean', 'f_ocean2'):
#             queue = cpu_kind
#         elif years <= 100:
#             queue = 'express'
# #             cpus = min(cpus, 8)
# #             cpu_kind = 'all'
#             cpu_kind = 'westmere'
#         else:
#             queue = 'medium'
#         
#         
#         ## super init
#         Job.init(self, job_name, JOB_MEMORY_GB, nodes, cpus, cpu_kind=cpu_kind, queue=queue)
        
        
        
        ## get output path
        output_path = os.path.abspath(self.output_dir)
        output_path = os.path.join(output_path, "") # ending with separator
        
        
        
        ## set model options
        opt = self.options
        
        model_parameters = np.array(model_parameters, dtype=np.float64)
        opt['/model/parameters'] = model_parameters
        opt['/model/parameters_file'] = os.path.join(output_path, 'model_parameter.txt')
        np.savetxt(opt['/model/parameters_file'], opt['/model/parameters'], fmt=MODEL_PARAMETERS_FORMAT_STRING)
        
        time_step_count = int(MODEL_TIME_STEP_SIZE_MAX / time_step)
        opt['/model/time_step_count'] = time_step_count
        opt['/model/time_step'] = 1 / time_step_count
        
        
        
        ## set metos3d options
        new_environment = opt['/job/queue'] in ('f_ocean', 'f_ocean2')
        if new_environment:
            METOS_PATH = METOS_PATH_2
        else:
            METOS_PATH = METOS_PATH_1
        opt['/metos3d/path'] = METOS_PATH
        opt['/metos3d/data_path'] = os.path.join(METOS_PATH, 'data/Metos3DData')
        opt['/metos3d/sim_file'] = os.path.join(METOS_PATH, 'simpack/metos3d-simpack-MITgcm-PO4-DOP.exe')
        opt['/metos3d/years'] = years
        opt['/metos3d/write_trajectory'] = write_trajectory
        if tolerance is not None:
            opt['/metos3d/tolerance'] = tolerance
        
        if write_trajectory:
            tracer_output_path = os.path.join(output_path, 'trajectory/')
            os.makedirs(tracer_output_path, exist_ok=True)
            opt['/metos3d/tracer_output_path'] = tracer_output_path
        
        opt['/metos3d/output_path'] = output_path
        opt['/metos3d/option_file'] = os.path.join(output_path, 'metos3d_options.txt')
        opt['/metos3d/debuglevel'] = 1
        opt['/metos3d/po4_output_filename'] = 'po4_output.petsc'
        opt['/metos3d/dop_output_filename'] = 'dop_output.petsc'
        
        if tracer_input_path is not None:
            opt['/metos3d/po4_input_filename'] = 'po4_input.petsc'
            opt['/metos3d/dop_input_filename'] = 'dop_input.petsc'
            
            tracer_input_path = os.path.relpath(tracer_input_path, start=output_path)
            
            os.symlink(os.path.join(tracer_input_path, opt['metos3d/po4_output_filename']), os.path.join(output_path, opt['/metos3d/po4_input_filename']))
            os.symlink(os.path.join(tracer_input_path, opt['metos3d/dop_output_filename']), os.path.join(output_path, opt['/metos3d/dop_input_filename']))
            
            opt['/metos3d/tracer_input_path'] = output_path
        
        model_parameters_string = ''
        model_parameters_len = len(model_parameters)
        for i in range(model_parameters_len):
            model_parameters_string += MODEL_PARAMETERS_FORMAT_STRING % model_parameters[i]
            if i < model_parameters_len - 1:
                model_parameters_string += ','
        
        opt['/metos3d/parameters_string'] = model_parameters_string
        
        
        
        ## write metos3d option file
        f = open(opt['/metos3d/option_file'], mode='w')

        f.write('# debug \n')
        f.write('-Metos3DDebugLevel                      %i \n\n' % opt['/metos3d/debuglevel'])
    
        f.write('# geometry \n')
        f.write('-Metos3DGeometryType                    Profile \n')
        f.write('-Metos3DProfileInputDirectory           %s/2.8/Geometry/ \n' % opt['/metos3d/data_path'])
        f.write('-Metos3DProfileIndexStartFile           gStartIndices.bin \n')
        f.write('-Metos3DProfileIndexEndFile             gEndIndices.bin \n\n')
    
        f.write('# bgc tracer \n')
        f.write('-Metos3DTracerCount                     2 \n')
        
        try:
            f.write('-Metos3DTracerInputDirectory            %s \n' % opt['/metos3d/tracer_input_path'])	
            f.write('-Metos3DTracerInitFile                  %s,%s \n' % (opt['/metos3d/po4_input_filename'], opt['/metos3d/dop_input_filename']))
        except KeyError:
            f.write('-Metos3DTracerInitValue                 2.17e+0,1.e-4 \n')
        
        try:
            f.write('-Metos3DTracerOutputDirectory           %s \n' % opt['/metos3d/tracer_output_path'])
        except KeyError:
            f.write('-Metos3DTracerOutputDirectory           %s \n' % opt['/metos3d/output_path'])
        
        f.write('-Metos3DTracerOutputFile                %s,%s \n\n' % (opt['/metos3d/po4_output_filename'], opt['/metos3d/dop_output_filename']))
        
        f.write('# bgc parameter \n')
        f.write('-Metos3DParameterCount                  7 \n')
        f.write('-Metos3DParameterValue                  %s \n\n' % opt['/metos3d/parameters_string'])
    
        f.write('# bgc boundary conditions \n')
        f.write('-Metos3DBoundaryConditionCount          2 \n')
        f.write('-Metos3DBoundaryConditionInputDirectory %s/2.8/Forcing/BoundaryCondition/ \n' % opt['/metos3d/data_path'])
        f.write('-Metos3DBoundaryConditionName           Latitude,IceCover \n')
        f.write('-Metos3DLatitudeCount                   1 \n')
        f.write('-Metos3DLatitudeFileFormat              latitude.petsc \n')
        f.write('-Metos3DIceCoverCount                   12 \n')
        f.write('-Metos3DIceCoverFileFormat              fice_$02d.petsc \n\n')
    
        f.write('# bgc domain conditions \n')
        f.write('-Metos3DDomainConditionCount            2 \n')
        f.write('-Metos3DDomainConditionInputDirectory   %s/2.8/Forcing/DomainCondition/ \n' % opt['/metos3d/data_path'])
        f.write('-Metos3DDomainConditionName             LayerDepth,LayerHeight \n')
        f.write('-Metos3DLayerDepthCount                 1 \n')
        f.write('-Metos3DLayerDepthFileFormat            z.petsc \n\n')
        f.write('-Metos3DLayerHeightCount                1 \n')
        f.write('-Metos3DLayerHeightFileFormat           dz.petsc \n')
    
        f.write('# transport \n')
        f.write('-Metos3DTransportType                   Matrix \n')
        f.write('-Metos3DMatrixInputDirectory            %s/2.8/Transport/Matrix5_4/1dt/ \n' % opt['/metos3d/data_path'])
        f.write('-Metos3DMatrixCount                     12 \n')
        f.write('-Metos3DMatrixExplicitFileFormat        Ae_$02d.petsc \n')
        f.write('-Metos3DMatrixImplicitFileFormat        Ai_$02d.petsc \n\n')
    
        f.write('# time stepping \n')
        f.write('-Metos3DTimeStepStart                   0.0 \n')
        f.write('-Metos3DTimeStepCount                   %i \n' % opt['/model/time_step_count'])
        f.write('-Metos3DTimeStep                        %.18f \n\n' % opt['/model/time_step'])
    
        f.write('# solver \n')
        f.write('-Metos3DSolverType                      Spinup \n')
        f.write('-Metos3DSpinupMonitor \n')
        try:
            f.write('-Metos3DSpinupTolerance                 %f \n' % opt['/metos3d/tolerance'])
        except KeyError:
            pass
        f.write('-Metos3DSpinupCount                     %i \n' % opt['/metos3d/years'])
    
        if opt['/metos3d/write_trajectory']:
            f.write('-Metos3DSpinupMonitorFileFormatPrefix   sp$0004d-,ts$0004d- \n')
            f.write('-Metos3DSpinupMonitorModuloStep         1,1 \n')
        
        util.io.flush_and_close(f)
        
        
        
        
        ## write job file
        if new_environment:
            run_command = 'mpirun -n {} -machinefile $PBS_NODEFILE -r rsh {} {} \n\n'
        else:
            run_command = 'mpirun -n {} -machinefile $PBS_NODEFILE {} {} \n\n'
        
        run_command = run_command.format(opt['/job/nodes'] * opt['/job/cpus'], opt['/metos3d/sim_file'], opt['/metos3d/option_file'])
#         
#         self.write_job_file(run_command, modules=('hdf5_1.8.8', 'petsc', 'python3'))
        self.write_job_file(run_command, modules=['petsc'])
        
        logger.debug('Job initialised.')