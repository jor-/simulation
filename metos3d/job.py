import os
import os.path as path
import time
import subprocess
import re
import numpy as np

import logging
logger = logging.getLogger(__name__)

import util.rzcluster.interact
from util.rzcluster.job import Job



class Metos3D_Job(Job):
    
    def __init__(self, output_path, pause_time_seconds=30, force_load=False):
        Job.__init__(self, output_path, pause_time_seconds, force_load)
    
    
    
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
    
    
    
    def init(self, model_parameters, years=1, tolerance=0, time_step_size=1, write_trajectory=False, tracer_input_path=None, nodes_max=None):
        from ndop.metos3d.constants import JOB_OPTIONS_FILENAME, JOB_MEMORY_GB, JOB_MIN_CPUS, JOB_NODES_MAX, JOB_NODES_LEFT_FREE, MODEL_PARAMETERS_FORMAT_STRING, MODEL_TIME_STEP_SIZE_MAX

        logger.debug('Initialising job.')
        
        
        ## check input
        if MODEL_TIME_STEP_SIZE_MAX % time_step_size != 0:
            raise ValueError('Wrong time_step_size passed. ' + str(MODEL_TIME_STEP_SIZE_MAX) + ' has to be divisible by time_step_size. But time_step_size is ' + str(time_step_size) + '.')
        
        
        
        ## get output path
        output_path = path.abspath(self.output_dir)
        output_path = path.join(output_path, "") # ending with separator
        
        
        
        ## set model options
        opt = self.options
        
        model_parameters = np.array(model_parameters, dtype=np.float64)
        opt['/model/parameters'] = model_parameters
        opt['/model/parameters_file'] = path.join(output_path, 'model_parameter.txt')
        np.savetxt(opt['/model/parameters_file'], opt['/model/parameters'], fmt=MODEL_PARAMETERS_FORMAT_STRING)
        
        opt['/model/time_step_size'] = time_step_size
        time_step_count = int(MODEL_TIME_STEP_SIZE_MAX / time_step_size)
        opt['/model/time_step_count'] = time_step_count
        opt['/model/time_step'] = 1 / time_step_count
        
        
        
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
            os.makedirs(tracer_output_path, exist_ok=True)
            opt['/metos3d/tracer_output_path'] = tracer_output_path
        
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
        
        f.flush()
        f.close()
        
        
        
        ## init job with best cpu configurations
        if years == 1:
            cpus_total_min = 1
            
            if nodes_max is not None:
                nodes_max = list(nodes_max)
                for i in range(len(nodes_max)):
                    nodes_max[i] = min(nodes_max[i], 1)
            else:
                nodes_max = (1,) * len(JOB_MIN_CPUS)
        else:
            cpus_total_min = JOB_MIN_CPUS
        
        (cpu_kind, nodes, cpus) = util.rzcluster.interact.wait_for_needed_resources(JOB_MEMORY_GB, cpus_total_min=cpus_total_min, nodes_max=nodes_max, node_left_free=JOB_NODES_LEFT_FREE)
        
        if cpu_kind == 'f_ocean':
            queue = 'f_ocean'
        elif years == 1:
            queue = 'express'
            cpus = min(cpus, 8)
            cpu_kind = 'all'
        else:
            queue = 'medium'
        
        job_name = '%i_%i' % (years, time_step_size)
        Job.init(self, nodes, cpus, JOB_MEMORY_GB, job_name, cpu_kind=cpu_kind, queue=queue)
        
        
        
        ## write job file
        run_command = 'mpirun -n %i -machinefile $PBS_NODEFILE %s %s \n\n' % (opt['/job/nodes'] * opt['/job/cpus'], opt['/metos3d/sim_file'], opt['/metos3d/option_file'])
        self.write_job_file(run_command, modules=('petsc3.3-gnu',))
        
        logger.debug('Job initialised.')