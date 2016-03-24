import os
import time
import subprocess
import re
import numpy as np

import simulation.model.constants

import util.batch.universal.system
import util.io.fs
import util.petsc.universal

import util.logging
logger = util.logging.logger


class Metos3D_Job(util.batch.universal.system.Job):


    @property
    def last_spinup_line(self):
        self.wait_until_finished()

        # 9.704s 0010 Spinup Function norm 2.919666257647e+00
        last_spinup_line = None
        with open(self.output_file) as f:
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
        from simulation.model.constants import METOS_T_DIM

        opt = self.options
        metos3d_opt_file = opt['/metos3d/option_file']
        with open(metos3d_opt_file) as metos3d_opt_file_object:
            metos3d_opt_lines = metos3d_opt_file_object.readlines()

        for metos3d_opt_line in metos3d_opt_lines:
            if re.search('Metos3DTimeStepCount', metos3d_opt_line) is not None:
                time_step_count = int(re.findall('\d+', metos3d_opt_line)[1])
                time_step = int(METOS_T_DIM / time_step_count)

        return time_step


    @property
    def metos3d_option_file(self):
        return self.options['/metos3d/option_file']
    

    @property
    def tracer_input_dir(self):
        opt = self.options

        try:
            tracer_input_dir = opt['/model/tracer_input_dir']
        except KeyError:
            tracer_input_dir = None

        return tracer_input_dir

    
    @property
    def tracer_output_dir(self):
        try:
            tracer_output_dir = self.options['/metos3d/tracer_output_path']
        except KeyError:
            tracer_output_dir = self.options['/metos3d/output_path']
        return tracer_output_dir
    
    
    @property
    def tracer_output_files(self):
        tracer_output_dir = self.tracer_output_dir
        tracer_output_files = [os.path.join(tracer_output_dir, tracer_output_filename) for tracer_output_filename in self.options['/metos3d/output_filenames']]
        return tracer_output_files
    
    
    @property
    def tracer_output_info_files(self):
        tracer_output_dir = self.tracer_output_dir
        tracer_output_info_files = [tracer_output_files + '.info' for tracer_output_files in self.tracer_output_files]
        return tracer_output_info_files

    

    def make_read_only_input(self, read_only=True):
        super().make_read_only_input(read_only=read_only)
        if read_only:
            util.io.fs.make_read_only(self.metos3d_option_file)

    def make_read_only_output(self, read_only=True):
        super().make_read_only_output(read_only=read_only)
        if read_only:
            for file in self.tracer_output_files:
                util.io.fs.make_read_only(file)
            for file in self.tracer_output_info_files:
                util.io.fs.make_read_only(file)



    @property
    def exit_code(self):
        ## get metos3d exit code
        exit_code = super().exit_code
        if exit_code != 0:
            return exit_code
        
        ## check if output file exists
        if self.output_file is not None and not os.path.exists(self.output_file):
            ValueError('Output file {} does not exist. The job is not finished'.format(self.output_file))

        ## check output file for errors
        IGNORE_ERRORS = ('Error_Path = ', 'cpuinfo: error while loading shared libraries: libgcc_s.so.1: cannot open shared object file: No such file or directory')
        output = self.output
        for ingore_error in IGNORE_ERRORS:
            output = output.replace(ingore_error, '')
        output = output.lower()
        if 'error' in output:
            return 255
        else:
            return 0
    

    def is_finished(self, check_exit_code=True):
        ## check if finished without exit code check
        if not super().is_finished(check_exit_code=False):
            return False
        
        ## ensure that output file exists for error check
        if check_exit_code and self.output_file is not None and not os.path.exists(self.output_file):
            return False

        ## check if finished with exit code check
        if check_exit_code and not super().is_finished(check_exit_code=check_exit_code):
            return False
            
        ## check if output file is completely written
        job_output = self.output
        if 'Metos3DFinal' in job_output:
            return True
        else:
            time.sleep(30)
            if self.output != job_output:
                return False
            else:
                raise util.batch.universal.system.JobError(self.id, self.output_dir, 'The job output file is not completely written!', job_output)
    
    

    def update_output_dir(self, new_output_path):
        opt = self.options
        old_output_path = opt['/metos3d/output_path']

        if old_output_path.endswith('/'):
            old_output_path = old_output_path[:-1]
        if new_output_path.endswith('/'):
            new_output_path = new_output_path[:-1]

        opt.replace_all_str_options(old_output_path, new_output_path)



    def write_job_file(self, model_name, model_parameters, years, tolerance=None, time_step=1, total_concentration_factor=1, write_trajectory=False, tracer_input_dir=None, job_setup=None):
        from simulation.model.constants import JOB_OPTIONS_FILENAME, JOB_MEMORY_GB, DATABASE_PARAMETERS_FORMAT_STRING,  METOS_T_DIM, METOS_DATA_DIR, METOS_SIM_FILE, MODEL_DEFAULT_INITIAL_CONCENTRATION, MODEL_TRACER

        logger.debug('Initialising job with years {}, tolerance {}, time step {}, total concentration factor {}, tracer_input_dir {} and job_setup {}.'.format(years, tolerance, time_step, total_concentration_factor, tracer_input_dir, job_setup))

        ## check input
        if not time_step in simulation.model.constants.METOS_TIME_STEPS:
            raise ValueError('Wrong time_step in model options. Time step has to be in {} .'.format(time_step, simulation.model.constants.METOS_TIME_STEPS))
        assert simulation.model.constants.METOS_T_DIM % time_step == 0

        if years < 0:
            raise ValueError('Years must be greater or equal 0, but it is {} .'.format(years))
        if tolerance < 0:
            raise ValueError('Tolerance must be greater or equal 0, but it is {} .'.format(tolerance))
        if total_concentration_factor < 0:
            raise ValueError('Total_concentration_factor must be greater or equal 0, but it is {} .'.format(total_concentration_factor))

        ## unpack job setup
        if job_setup is not None:
            try:
                job_name = job_setup['name']
            except KeyError:
                job_name = 'Metos3D'
            try:
                nodes_setup = job_setup['nodes_setup']
            except KeyError:
                nodes_setup = None
        else:
            job_name = ''
            nodes_setup = None

        ## prepare job name
        if len(job_name) > 0:
            job_name += '_'
        job_name += '{}_{}_{}'.format(model_name, years, time_step)

        ## use best node setup if no node setup passed
        if nodes_setup is None:
            nodes_setup = util.batch.universal.system.NodeSetup()

        ## check/set memory
        if nodes_setup.memory is None:
            nodes_setup.memory = simulation.model.constants.JOB_MEMORY_GB
        elif nodes_setup.memory < simulation.model.constants.JOB_MEMORY_GB:
            logger.warn('The chosen memory {} is below the needed memory {}. Changing to needed memory.'.format(nodes_setup.memory, simulation.model.constants.JOB_MEMORY_GB))
            nodes_setup.memory = simulation.model.constants.JOB_MEMORY_GB

        ## check/set walltime
        sec_per_year = np.exp(- (nodes_setup.nodes * nodes_setup.cpus) / 40) * 30 + 1.5
        sec_per_year /= time_step**(1/2)
        estimated_walltime_hours = np.ceil(years * sec_per_year / 60**2) + 1
        if nodes_setup.walltime is None:
            nodes_setup.walltime = estimated_walltime_hours
        else:
            if nodes_setup.walltime < estimated_walltime_hours:
                logger.debug('The chosen walltime {} for the job with {} years, {} nodes and {} cpus is below the estimated walltime {}.'.format(nodes_setup.walltime, years, nodes_setup.nodes, nodes_setup.cpus, estimated_walltime_hours))

        ## check/set min cpus
        if nodes_setup.total_cpus_min is None:
            nodes_setup.total_cpus_min = min(int(np.ceil(years/20)), 32)

        ## check/set max nodes
        if nodes_setup.nodes_max is None and years <= 1:
            nodes_setup.nodes_max = 1


        ## init job
        super().init_job_file(job_name, nodes_setup)


        ## get output dir
        output_dir = self.output_dir
        output_dir_not_expanded = os.path.join(self.output_dir_not_expanded, '') # ending with separator


        ## set model options
        opt = self.options

        model_parameters = np.asarray(model_parameters, dtype=np.float64)
        assert len(model_parameters) == 7
        
        opt['/model/tracer'] = MODEL_TRACER[model_name]
        
        opt['/model/total_concentration_factor'] = total_concentration_factor
        opt['/model/parameters'] = model_parameters
        
        time_steps_per_year = int(METOS_T_DIM / time_step)
        opt['/model/time_step_multiplier'] = time_step
        opt['/model/time_steps_per_year'] = time_steps_per_year
        opt['/model/time_step'] = 1 / time_steps_per_year
        

        ## set metos3d options
        opt['/metos3d/data_path'] = METOS_DATA_DIR
        opt['/metos3d/sim_file'] = METOS_SIM_FILE
        opt['/metos3d/years'] = years
        opt['/metos3d/write_trajectory'] = write_trajectory
        if tolerance is not None:
            opt['/metos3d/tolerance'] = tolerance

        if write_trajectory:
            tracer_output_dir = os.path.join(output_dir, 'trajectory/')
            os.makedirs(tracer_output_dir, exist_ok=True)
            tracer_output_dir_not_expanded = os.path.join(output_dir_not_expanded, 'trajectory/')
            opt['/metos3d/tracer_output_path'] = tracer_output_dir_not_expanded

        opt['/metos3d/output_path'] = output_dir_not_expanded
        opt['/metos3d/option_file'] = os.path.join(output_dir_not_expanded, 'metos3d_options.txt')
        opt['/metos3d/debuglevel'] = 1
        opt['/metos3d/output_filenames'] = ['{}_output.petsc'.format(tracer) for tracer in opt['/model/tracer']]

        if tracer_input_dir is None:
            initial_concentration = MODEL_DEFAULT_INITIAL_CONCENTRATION[model_name] * total_concentration_factor
            opt['/model/initial_concentrations'] = initial_concentration
        else:
            opt['/model/tracer_input_dir'] = tracer_input_dir
            opt['/metos3d/tracer_input_dir'] = output_dir_not_expanded

            opt['/metos3d/input_filenames'] = ['{}_input.petsc'.format(tracer) for tracer in opt['/model/tracer']]
            
            if total_concentration_factor == 1:
                tracer_input_dir = os.path.relpath(tracer_input_dir, start=output_dir)
                for i in range(len(opt['/model/tracer'])):
                    tracer_input_base_file = os.path.join(tracer_input_dir, opt['metos3d/output_filenames'][i])
                    tracer_input_result_file = os.path.join(output_dir, opt['/metos3d/input_filenames'][i])
                    os.symlink(tracer_input_base_file, tracer_input_result_file)
            else:
                for i in range(len(opt['/model/tracer'])):
                    tracer_input_base_file = os.path.join(tracer_input_dir, opt['metos3d/output_filenames'][i])
                    tracer_input_result_file = os.path.join(output_dir, opt['/metos3d/input_filenames'][i])
                    tracer_input = util.petsc.universal.load_petsc_vec_to_numpy_array(tracer_input_base_file)
                    tracer_input = tracer_input * total_concentration_factor
                    util.petsc.universal.save_numpy_array_to_petsc_vec(tracer_input_result_file, tracer_input)
        

        model_parameters_string = ','.join(map(lambda f: DATABASE_PARAMETERS_FORMAT_STRING.format(f), model_parameters))
        opt['/metos3d/parameters_string'] = model_parameters_string


        ## write metos3d option file
        f = open(opt['/metos3d/option_file'], mode='w')

        f.write('# debug \n')
        f.write('-Metos3DDebugLevel                      {:d} \n\n'.format(opt['/metos3d/debuglevel']))

        f.write('# geometry \n')
        f.write('-Metos3DGeometryType                    Profile \n')
        f.write('-Metos3DProfileInputDirectory           {}/Geometry/ \n'.format(opt['/metos3d/data_path']))
        f.write('-Metos3DProfileIndexStartFile           gStartIndices.bin \n')
        f.write('-Metos3DProfileIndexEndFile             gEndIndices.bin \n\n')

        f.write('# bgc tracer \n')
        f.write('-Metos3DTracerCount                     2 \n')

        try:
            f.write('-Metos3DTracerInputDirectory            {} \n'.format(opt['/metos3d/tracer_input_dir']))
            f.write('-Metos3DTracerInitFile                  {} \n'.format(','.join(map(str, opt['/metos3d/input_filenames']))))
        except KeyError:
            f.write('-Metos3DTracerInitValue                 {},{} \n'.format(*opt['/model/initial_concentrations']))

        try:
            f.write('-Metos3DTracerOutputDirectory           {} \n'.format(opt['/metos3d/tracer_output_path']))
        except KeyError:
            f.write('-Metos3DTracerOutputDirectory           {} \n'.format(opt['/metos3d/output_path']))

        f.write('-Metos3DTracerOutputFile                {} \n\n'.format(','.join(map(str, opt['/metos3d/output_filenames']))))

        f.write('# bgc parameter \n')
        f.write('-Metos3DParameterCount                  {:d} \n'.format(len(opt['/model/parameters'])))
        f.write('-Metos3DParameterValue                  {} \n\n'.format(opt['/metos3d/parameters_string']))

        f.write('# bgc boundary conditions \n')
        f.write('-Metos3DBoundaryConditionCount          2 \n')
        f.write('-Metos3DBoundaryConditionInputDirectory {}/Forcing/BoundaryCondition/ \n'.format(opt['/metos3d/data_path']))
        f.write('-Metos3DBoundaryConditionName           Latitude,IceCover \n')
        f.write('-Metos3DLatitudeCount                   1 \n')
        f.write('-Metos3DLatitudeFileFormat              latitude.petsc \n')
        f.write('-Metos3DIceCoverCount                   12 \n')
        f.write('-Metos3DIceCoverFileFormat              fice_$02d.petsc \n\n')

        f.write('# bgc domain conditions \n')
        f.write('-Metos3DDomainConditionCount            2 \n')
        f.write('-Metos3DDomainConditionInputDirectory   {}/Forcing/DomainCondition/ \n'.format(opt['/metos3d/data_path']))
        f.write('-Metos3DDomainConditionName             LayerDepth,LayerHeight \n')
        f.write('-Metos3DLayerDepthCount                 1 \n')
        f.write('-Metos3DLayerDepthFileFormat            z.petsc \n\n')
        f.write('-Metos3DLayerHeightCount                1 \n')
        f.write('-Metos3DLayerHeightFileFormat           dz.petsc \n')

        f.write('# transport \n')
        f.write('-Metos3DTransportType                   Matrix \n')
        f.write('-Metos3DMatrixInputDirectory            {}/Transport/Matrix5_4/{:d}dt/ \n'.format(opt['/metos3d/data_path'], opt['/model/time_step_multiplier']))
        f.write('-Metos3DMatrixCount                     12 \n')
        f.write('-Metos3DMatrixExplicitFileFormat        Ae_$02d.petsc \n')
        f.write('-Metos3DMatrixImplicitFileFormat        Ai_$02d.petsc \n\n')

        f.write('# time stepping \n')
        f.write('-Metos3DTimeStepStart                   0.0 \n')
        f.write('-Metos3DTimeStepCount                   {:d} \n'.format(opt['/model/time_steps_per_year']))
        f.write('-Metos3DTimeStep                        {:.18f} \n\n'.format(opt['/model/time_step']))

        f.write('# solver \n')
        f.write('-Metos3DSolverType                      Spinup \n')
        f.write('-Metos3DSpinupMonitor \n')
        try:
            f.write('-Metos3DSpinupTolerance                 {:f} \n'.format(opt['/metos3d/tolerance']))
        except KeyError:
            pass
        f.write('-Metos3DSpinupCount                     {:d} \n'.format(opt['/metos3d/years']))

        if opt['/metos3d/write_trajectory']:
            f.write('-Metos3DSpinupMonitorFileFormatPrefix   sp$0004d-,ts$0004d- \n')
            f.write('-Metos3DSpinupMonitorModuloStep         1,1 \n')

        util.io.fs.flush_and_close(f)


        ## write job file
        run_command = '{} {} \n'.format(opt['/metos3d/sim_file'], opt['/metos3d/option_file'])
        super().write_job_file(run_command, modules=['intel', 'intelmpi', 'petsc'])

        logger.debug('Job initialised.')
