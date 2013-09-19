import os
import os.path as path
import time
import subprocess
import re
import numpy as np

import util.rzcluster

from util.debug import Debug
from util.options import Options


class Job(Debug):
    
    def __init__(self, debug_level=0, required_debug_level=1):
        Debug.__init__(self, debug_level, required_debug_level-1, 'ndop.metos3d.job: ')
    
    def __del__(self):
        self.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        self.close()
    
    
    
    @property
    def options(self):
        try:
            return self.__options
        except AttributeError:
            raise Exception("Job is not initialised!")
    
    @property
    def id(self):
        opt = self.options
        
        try:
            return opt['/job/id']
        except KeyError:
            raise Exception("Job is not started!")
    
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
#         opt = self.options
#         output_file = opt['/job/output_file']
#         
#         # 9.704s 0010 Spinup Function norm 2.919666257647e+00
#         spinup_last_line = None
#         with open(output_file) as f:
#             for line in f.readlines():
#                 if 'Spinup Function norm' in line:
#                     spinup_last_line = line
#         
#         if spinup_last_line is not None:
#             spinup_last_line = spinup_last_line.strip()
#             spinup_last_year_str = spinup_last_line.split()[1]
#             spinup_last_year = int(spinup_last_year_str) + 1
#         else:
#             spinup_last_year = 0
#         
#         return spinup_last_year
    
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
        from ndop.metos3d.constants import MODEL_TIME_STEP_SIZE_MAX
        
        opt = self.options
        metos3d_opt_file = opt['/metos3d/option_file']
        with open(metos3d_opt_file) as metos3d_opt_file_object:
            metos3d_opt_lines = metos3d_opt_file_object.readlines()
        
        for metos3d_opt_line in metos3d_opt_lines:
            if re.search('Metos3DTimeStepCount', metos3d_opt_line) is not None:
                time_step_count = int(re.findall('\d+', metos3d_opt_line)[1])
                time_step = MODEL_TIME_STEP_SIZE_MAX / time_step_count
                return time_step
    
    
    
    def load(self, option_file):
        from ndop.metos3d.constants import JOB_OPTIONS_FILENAME
        
        if path.isdir(option_file):
            option_file = path.join(option_file, JOB_OPTIONS_FILENAME)
        
        self.print_debug_inc(('Loading job from file"', option_file, '".'))
        
        try:
            opt = Options(option_file, mode='r+', debug_level=self.debug_level, required_debug_level=self.required_debug_level+1)
        except (OSError, IOError):
            opt = Options(option_file, mode='r', debug_level=self.debug_level, required_debug_level=self.required_debug_level+1)
        
        self.__options = opt
        
        self.print_debug_dec(('Job loaded from file"', option_file, '".'))
    
    
    def make_readonly(self):
        self.options.make_readonly()
    
    
    def get_tracer_input_path(self):
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
       
    
#     def initialise(self, model_parameters, output_path=os.getcwd(), years=1, tolerance=0, time_step_size=1, walltime_hours=240, cpu_kind='westmere', nodes=1, cpus=1, write_trajectory=False, tracer_input_path=None, pause_time_seconds=10):
    def initialise(self, model_parameters, output_path=os.getcwd(), years=1, tolerance=0, time_step_size=1, cpu_kind='f_ocean', queue='f_ocean', nodes=1, cpus=1, write_trajectory=False, tracer_input_path=None, pause_time_seconds=10):
        from ndop.metos3d.constants import JOB_OPTIONS_FILENAME, JOB_MEMORY_GB, MODEL_PARAMETERS_FORMAT_STRING, MODEL_TIME_STEP_SIZE_MAX
        from util.rzcluster_constants import QUEUES

        self.print_debug_inc('Initialising job.')
        
        ## check input
        if MODEL_TIME_STEP_SIZE_MAX % time_step_size != 0:
            raise ValueError('Wrong time_step_size passed. ' + str(MODEL_TIME_STEP_SIZE_MAX) + ' has to be divisible by time_step_size. But time_step_size is ' + str(time_step_size) + '.')
        
        if queue not in QUEUES:
            raise ValueError('Unknown queue ' + str(queue) + '.')
        
        
        
        ## create option file
        output_path = path.abspath(output_path)
        output_path = path.join(output_path, "") # ending with separator
#         if not path.exists(output_path):
#             os.makedirs(output_path)
        os.makedirs(output_path, exist_ok=True)
        
        opt = Options(path.join(output_path, JOB_OPTIONS_FILENAME), mode='w-', debug_level=self.debug_level, required_debug_level=self.required_debug_level+1)
        self.__options = opt
        
        
        
        ## set model options
        model_parameters = np.array(model_parameters, dtype=np.float64)
        opt['/model/parameters'] = model_parameters
        opt['/model/parameters_file'] = path.join(output_path, 'model_parameter.txt')
        np.savetxt(opt['/model/parameters_file'], opt['/model/parameters'], fmt=MODEL_PARAMETERS_FORMAT_STRING)
        
        opt['/model/time_step_size'] = time_step_size
        time_step_count = int(MODEL_TIME_STEP_SIZE_MAX / time_step_size)
        opt['/model/time_step_count'] = time_step_count
        opt['/model/time_step'] = 1 / time_step_count
        
        
        
        ## set job options
#         opt['/job/walltime_hours'] = walltime_hours
        opt['/job/nodes'] = nodes
        opt['/job/cpus'] = cpus
        opt['/job/memory_gb'] = JOB_MEMORY_GB
        
#         if cpu_kind == 'barcelona':
#             cpu_kind = 'f_ocean'
#         
#         if cpu_kind == 'opteron':
#             cpu_kind = 'all'
        opt['/job/cpu_kind'] = cpu_kind
        
        if queue == 'f_ocean':
            walltime_hours = 240
        elif queue == 'express':
            walltime_hours = 3
        elif queue == 'small':
            walltime_hours = 24
        elif queue == 'medium':
            walltime_hours = 240
        elif queue == 'long':
            walltime_hours = 480
        elif queue == 'para_low':
            walltime_hours = 1000
            
        opt['/job/walltime_hours'] = walltime_hours
#         
#         
#         if cpu_kind == 'f_ocean':
#             queue = 'f_ocean'
#         elif cpu_kind == 'westmere' or cpu_kind == 'all':
#             if walltime_hours <= 3:
#                 queue = 'express'
#             elif walltime_hours <= 24:
#                 queue = 'small'
#             elif walltime_hours <= 240:
#                 queue = 'medium'
#             elif walltime_hours <= 480:
#                 queue = 'long'
#             elif walltime_hours <= 1000:
#                 queue = 'para_low'
#             else:
#                 raise ValueError('Walltime hours ' + str(walltime_hours) + ' to long.')
#         else:
#             raise ValueError('Unknown cpu_kind ' + str(cpu_kind) + '.')
        
            
        opt['/job/queue'] = queue
        
        opt['/job/output_path'] = path.dirname(output_path)
        opt['/job/option_file'] = path.join(output_path, 'job_options.txt')
        opt['/job/output_file'] = path.join(output_path, 'job_output.txt')
        opt['/job/id_file'] = path.join(output_path, 'job_id.txt')
        opt['/job/finished_file'] = path.join(output_path, 'finished.txt')
        
        opt['/job/pause_time_seconds'] = pause_time_seconds
        
        
        
        ## set metos3d options
        metos3d_path = '/work_j2/sunip229/NDOP/metos3d/v0.2'
        opt['/metos3d/path'] = metos3d_path
        opt['/metos3d/data_path'] = path.join(metos3d_path, 'data/Metos3DData')
        opt['/metos3d/sim_file'] = path.join(metos3d_path, 'simpack/metos3d-simpack-MITgcm-PO4-DOP.exe')
        opt['/metos3d/years'] = years
        opt['/metos3d/write_trajectory'] = write_trajectory
        if tolerance is not None:
            opt['/metos3d/tolerance'] = tolerance
        
        if write_trajectory:
            tracer_output_path = path.join(output_path, 'trajectory/')
#             if not path.exists(tracer_output_path):
#                 os.makedirs(tracer_output_path)
            os.makedirs(tracer_output_path, exist_ok=True)
            opt['/metos3d/tracer_output_path'] = tracer_output_path
        
        opt['/job/name'] = '%i_%i' % (years, time_step_size)
        
        opt['/metos3d/output_path'] = output_path
        opt['/metos3d/option_file'] = path.join(output_path, 'metos3d_options.txt')
        opt['/metos3d/debuglevel'] = 1
        opt['/metos3d/po4_output_filename'] = 'po4_output.petsc'
        opt['/metos3d/dop_output_filename'] = 'dop_output.petsc'
        
        if tracer_input_path is not None:
            opt['/metos3d/po4_input_filename'] = 'po4_input.petsc'
            opt['/metos3d/dop_input_filename'] = 'dop_input.petsc'
            
            tracer_input_path = path.relpath(tracer_input_path, start=output_path)
            
            os.symlink(path.join(tracer_input_path, opt['metos3d/po4_output_filename']), path.join(output_path, opt['/metos3d/po4_input_filename']))
            os.symlink(path.join(tracer_input_path, opt['metos3d/dop_output_filename']), path.join(output_path, opt['/metos3d/dop_input_filename']))
            
            opt['/metos3d/tracer_input_path'] = output_path
        
        model_parameters_string = ''
        model_parameters_len = len(model_parameters)
        for i in range(model_parameters_len):
            model_parameters_string += MODEL_PARAMETERS_FORMAT_STRING % model_parameters[i]
            if i < model_parameters_len - 1:
                model_parameters_string += ','
        
#         model_parameters_last_index = len(model_parameters) - 1
#         for i in range(model_parameters_last_index):
#             model_parameters_string += MODEL_PARAMETERS_FORMAT_STRING + ', ' % model_parameters[i]
#         model_parameters_string += MODEL_PARAMETERS_FORMAT_STRING % model_parameters[model_parameters_last_index]
        opt['/metos3d/parameters_string'] = model_parameters_string
        

        
        
        ## write job file
        f = open(opt['/job/option_file'], mode='w')

        f.write('#!/bin/bash \n\n')
        
        f.write('#PBS -N %s \n' % opt['/job/name'])
        f.write('#PBS -j oe \n')
        f.write('#PBS -o %s \n' % opt['/job/output_file'])
        
        try:
            f.write('#PBS -l walltime=%02i:00:00 \n' % opt['/job/walltime_hours'])
        except KeyError:
            pass
        
        f.write('#PBS -l select=%i:%s=true:ncpus=%i:mpiprocs=%i:mem=%igb \n' % (opt['/job/nodes'], opt['/job/cpu_kind'], opt['/job/cpus'], opt['/job/cpus'], opt['/job/memory_gb']))
        f.write('#PBS -q %s \n\n' % opt['/job/queue'])
        
        f.write('. /usr/local/Modules/3.2.6/init/bash \n\n')
        
        f.write('module load petsc3.3-gnu \n')
        f.write('module list \n\n')
        
        f.write('cd $PBS_O_WORKDIR \n\n')
        
        f.write('mpirun -n %i -machinefile $PBS_NODEFILE %s %s \n\n' % (opt['/job/nodes'] * opt['/job/cpus'], opt['/metos3d/sim_file'], opt['/metos3d/option_file']))
        
        f.write('touch %s \n\n' % opt['/job/finished_file'])
        
        f.write('qstat -f $PBS_JOBID \n')
        f.write('exit \n')
        
        f.flush()
        f.close()
        
        
        
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
        
        f.flush()
        f.close()
        
        self.print_debug_dec('Job initialised.')
    
    
    
    def initialise_with_best_configuration(self, model_parameters, output_path=os.getcwd(), years=1, tolerance=0, time_step_size=1, write_trajectory=False, tracer_input_path=None, pause_time_seconds=10):
        
        from ndop.metos3d.constants import JOB_MEMORY_GB, JOB_MIN_CPUS
        
        self.print_debug_inc('Getting best job configutrations.')
        
        if years == 1:
            max_nodes = 1
        else:
            max_nodes = float('inf')
        
        nodes = 0
        cpus = 0
        resources_free = False
        while not resources_free:
            (cpu_kind, nodes, cpus) =  util.rzcluster.get_best_cpu_configurations(JOB_MEMORY_GB * 1024, nodes_ub=max_nodes, debug_level=self.debug_level, required_debug_level=self.required_debug_level+1)
            resources_free = (nodes * cpus >= JOB_MIN_CPUS) or (write_trajectory and nodes >= 1)
            if not resources_free:
                self.print_debug('No enough resources free. Waiting.')
                time.sleep(60)
        
        if cpu_kind == 'f_ocean':
            queue = 'f_ocean'
        elif years == 1:
            queue = 'express'
            cpus = min(cpus, 8)
            cpu_kind = 'all'
        else:
            queue = 'medium'
        
#         walltime_hours = 240
#         self.initialise(model_parameters, output_path=output_path, walltime_hours=walltime_hours, cpu_kind=cpu_kind, nodes=nodes, cpus=cpus, years=years, tolerance=tolerance, time_step_size=time_step_size, write_trajectory=write_trajectory, tracer_input_path=tracer_input_path, pause_time_seconds=pause_time_seconds)
        self.initialise(model_parameters, output_path=output_path, cpu_kind=cpu_kind, queue=queue, nodes=nodes, cpus=cpus, years=years, tolerance=tolerance, time_step_size=time_step_size, write_trajectory=write_trajectory, tracer_input_path=tracer_input_path, pause_time_seconds=pause_time_seconds)
        
        self.print_debug_dec('Got best job configurations.')
    
        
        
    def start(self):
        opt = self.options
        
        job_id = util.rzcluster.start_job(opt['/job/option_file'], debug_level=self.debug_level, required_debug_level=self.required_debug_level+1)
        
        opt['/job/id'] = job_id
        
        with open(opt['/job/id_file'], "w") as job_id_file:
            job_id_file.write(job_id)
        
    
    
    
    def is_finished(self):
        job_id = self.id
        
        is_finished = util.rzcluster.is_job_finished(job_id, debug_level=self.debug_level, required_debug_level=self.required_debug_level+1)
        
        return is_finished
    
    
    
    def wait_until_finished(self):
        job_id = self.id
        opt = self.options
        pause_time_seconds = opt['/job/pause_time_seconds']
        
        util.rzcluster.wait_until_job_finished(job_id, pause_time_seconds=opt['/job/pause_time_seconds'], debug_level=self.debug_level, required_debug_level=self.required_debug_level+1)
    
    
    def close(self):
        try:
            options = self.__options
        except AttributeError:
            options = None
        
        if options is not None:
            options.close()
    