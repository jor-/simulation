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

    ## run options
    
    @property
    def last_spinup_line(self):
        self.wait_until_finished()

        # 9.704s 0010 Spinup Function norm 2.919666257647e+00
        # 9.704s 0010 Spinup Function norm 2.919666257647e+00 7.012035082243e+06
        
        search_str = 'Spinup Function norm'
        last_spinup_line = None
        output = self.output
        for line in output.splitlines():
            if search_str in line:
                last_spinup_line = line
        
        if last_spinup_line is None:
            error_message = 'In job output is no "{}" line.'.format(search_str)
            raise util.batch.universal.system.JobError(self, error_message, include_ouput=True)

        return last_spinup_line


    @property
    def last_year(self):
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
        opt = self.options
        metos3d_opt_file = opt['/metos3d/option_file']
        with open(metos3d_opt_file) as metos3d_opt_file_object:
            metos3d_opt_lines = metos3d_opt_file_object.readlines()

        for metos3d_opt_line in metos3d_opt_lines:
            if re.search('Metos3DTimeStepCount', metos3d_opt_line) is not None:
                time_step_count = int(re.findall('\d+', metos3d_opt_line)[1])
                time_step = int(simulation.model.constants.METOS_T_DIM / time_step_count)

        return time_step


    ## files
    
    @property
    def metos3d_option_file(self):
        return self.options['/metos3d/option_file']
    
    
    @property
    def model_tracer_input_files(self):
        return self.options['/model/tracer_input_files']

    @property
    def tracer_input_dir(self):
        try:
            tracer_input_dir = self.options['/metos3d/tracer_input_dir']
        except KeyError:
            tracer_input_dir = None

        return tracer_input_dir

    
    @property
    def tracer_input_files(self):
        try:
            tracer_input_dir = self.options['/metos3d/tracer_input_dir']
        except KeyError:
            tracer_input_files = []
        else:
            tracer_input_files = [os.path.join(tracer_input_dir, tracer_input_filename) for tracer_input_filename in self.options['/metos3d/tracer_input_filenames']]
        return tracer_input_files
    
    
    @property
    def tracer_input_info_files(self):
        tracer_input_info_files = [tracer_input_file + '.info' for tracer_input_file in self.tracer_input_files]
        return tracer_input_info_files
    
    
    @property
    def tracer_output_dir(self):
        return self.options['/metos3d/tracer_output_dir']
    
    
    @property
    def tracer_output_files(self):
        tracer_output_dir = self.tracer_output_dir
        tracer_output_files = [os.path.join(tracer_output_dir, tracer_output_filename) for tracer_output_filename in self.options['/metos3d/tracer_output_filenames']]
        return tracer_output_files
    
    
    @property
    def tracer_output_info_files(self):
        tracer_output_info_files = [tracer_output_file + '.info' for tracer_output_file in self.tracer_output_files]
        return tracer_output_info_files

    

    def make_read_only_input(self, read_only=True):
        super().make_read_only_input(read_only=read_only)
        if read_only:
            util.io.fs.make_read_only(self.metos3d_option_file)
            for file in self.tracer_input_files:
                util.io.fs.make_read_only(file)


    def make_read_only_output(self, read_only=True):
        super().make_read_only_output(read_only=read_only)
        if read_only:
            for file in self.tracer_output_files:
                util.io.fs.make_read_only(file)
            for file in self.tracer_output_info_files:
                util.io.fs.make_read_only(file)


    ## exit code and is finished

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
                raise util.batch.universal.system.JobError(self, 'The job output file is not completely written!', job_output)


    ## write job file

    def write_job_file(self, model_name, model_parameters, years, tolerance=None, time_step=1, initial_constant_concentrations=None, tracer_input_files=None, total_concentration_factor=1, write_trajectory=False, job_options=None):

        logger.debug('Initialising job with model {}, parameters {},  years {}, tolerance {}, time step {}, initial_constant_concentrations {}, tracer_input_files {}, total concentration factor {} and job_options {}.'.format(model_name, model_parameters, years, tolerance, time_step, initial_constant_concentrations, tracer_input_files, total_concentration_factor, job_options))

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

        if initial_constant_concentrations is not None and tracer_input_files is not None:
            raise ValueError('You can not set the initial concentration and the tracer input files simultaneously.')
        
        number_of_tracers = len(simulation.model.constants.MODEL_TRACER[model_name])
        if initial_constant_concentrations is not None:
            initial_constant_concentrations = np.asanyarray(initial_constant_concentrations)
            if len(initial_constant_concentrations) != number_of_tracers:
                raise ValueError('The initial concentration must be {} values for model {}, but it is {}.'.format(number_of_tracers, model_name, initial_constant_concentrations))
        if tracer_input_files is not None:
            if len(tracer_input_files) != number_of_tracers:
                raise ValueError('The tracer input files must be {} files for model {}, but it is {}.'.format(number_of_tracers, model_name, tracer_input_files))
                

        ## unpack job setup
        if job_options is not None:
            try:
                job_name = job_options['name']
            except KeyError:
                job_name = 'Metos3D'
            try:
                nodes_setup = job_options['nodes_setup']
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
        sec_per_year = np.exp(- (nodes_setup.nodes * nodes_setup.cpus) / (6*16)) * 10 + 2.5
        sec_per_year /= time_step**(1/2)
        estimated_walltime_hours = np.ceil(years * sec_per_year / 60**2)
        logger.debug('The estimated walltime for {} nodes with {} cpus, {} years and time step {} is {} hours.'.format(nodes_setup.nodes, nodes_setup.cpus, years, time_step, estimated_walltime_hours))
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

        opt['/model/name'] = model_name
        opt['/model/tracer'] = simulation.model.constants.MODEL_TRACER[model_name]
        
        time_steps_per_year = int(simulation.model.constants.METOS_T_DIM / time_step)
        opt['/model/time_step'] = 1 / time_steps_per_year
        opt['/model/time_steps_per_year'] = time_steps_per_year
        opt['/model/time_step_multiplier'] = time_step

        opt['/model/spinup/years'] = years
        if tolerance is not None:
            opt['/model/spinup/tolerance'] = tolerance
        
        ## set metos3d files and dirs
        opt['/metos3d/data_dir'] = simulation.model.constants.METOS_DATA_DIR_ENV
        opt['/metos3d/sim_file'] = simulation.model.constants.METOS_SIM_FILE_ENV.format(model_name=model_name, METOS3D_DIR='{METOS3D_DIR}')
        opt['/metos3d/write_trajectory'] = write_trajectory

        if not write_trajectory:
            opt['/metos3d/tracer_output_dir'] = output_dir_not_expanded
        else:
            tracer_output_dir = os.path.join(output_dir, 'trajectory/')
            os.makedirs(tracer_output_dir, exist_ok=True)
            tracer_output_dir_not_expanded = os.path.join(output_dir_not_expanded, 'trajectory/')
            opt['/metos3d/tracer_output_dir'] = tracer_output_dir_not_expanded

        opt['/metos3d/output_dir'] = output_dir_not_expanded
        opt['/metos3d/option_file'] = os.path.join(output_dir_not_expanded, 'metos3d_options.txt')
        opt['/metos3d/debuglevel'] = 1
        opt['/metos3d/tracer_output_filenames'] = ['{}_output.petsc'.format(tracer) for tracer in opt['/model/tracer']]
        
        ## tracer_input_files
        if tracer_input_files is not None:
            opt['/model/tracer_input_files'] = tracer_input_files
            
            opt['/metos3d/tracer_input_dir'] = output_dir_not_expanded
            opt['/metos3d/tracer_input_filenames'] = ['{}_input.petsc'.format(tracer) for tracer in opt['/model/tracer']]
            
            for i in range(len(opt['/model/tracer'])):
                tracer_input_base_file = os.path.expanduser(os.path.expandvars(tracer_input_files[i]))
                tracer_input_result_file = os.path.join(output_dir, opt['/metos3d/tracer_input_filenames'][i])
            
                if total_concentration_factor == 1:
                    tracer_input_base_file = os.path.relpath(tracer_input_base_file, start=output_dir)
                    os.symlink(tracer_input_base_file, tracer_input_result_file)
                else:
                    tracer_input = util.petsc.universal.load_petsc_vec_to_numpy_array(tracer_input_base_file)
                    tracer_input = tracer_input * total_concentration_factor
                    util.petsc.universal.save_numpy_array_to_petsc_vec(tracer_input_result_file, tracer_input)

        ## initial_constant_concentrations
        else:
            if initial_constant_concentrations is None:
                initial_constant_concentrations = simulation.model.constants.MODEL_DEFAULT_INITIAL_CONCENTRATION[model_name]
            initial_constant_concentrations = initial_constant_concentrations * total_concentration_factor
            opt['/model/initial_constant_concentrations'] = initial_constant_concentrations
            
            initial_constant_concentrations_string = ','.join(map(str, initial_constant_concentrations))
            opt['/metos3d/initial_constant_concentrations_string'] = initial_constant_concentrations_string
        
        ## model parameter
        model_parameters = np.asarray(model_parameters, dtype=np.float64)
        opt['/model/parameters'] = model_parameters
        
        model_parameters_string = ','.join(map(lambda f: simulation.model.constants.DATABASE_PARAMETERS_FORMAT_STRING.format(f), model_parameters))
        opt['/metos3d/parameters_string'] = model_parameters_string
        
        ## initial concentrations

        ## write metos3d option file
        f = open(opt['/metos3d/option_file'], mode='w')

        f.write('# debug \n')
        f.write('-Metos3DDebugLevel                      {:d} \n\n'.format(opt['/metos3d/debuglevel']))

        f.write('# geometry \n')
        f.write('-Metos3DGeometryType                    Profile \n')
        f.write('-Metos3DProfileInputDirectory           {}/Geometry/ \n'.format(opt['/metos3d/data_dir']))
        # f.write('-Metos3DProfileIndexStartFile           gStartIndices.bin \n')
        # f.write('-Metos3DProfileIndexEndFile             gEndIndices.bin \n\n')
        f.write('-Metos3DProfileMaskFile                 landSeaMask.petsc \n')
        f.write('-Metos3DProfileVolumeFile               volumes.petsc \n\n')

        f.write('# bgc tracer \n')
        f.write('-Metos3DTracerCount                     {:d} \n'.format(len(opt['/model/tracer'])))

        try:
            f.write('-Metos3DTracerInputDirectory            {} \n'.format(opt['/metos3d/tracer_input_dir']))
            f.write('-Metos3DTracerInitFile                  {} \n'.format(','.join(map(str, opt['/metos3d/tracer_input_filenames']))))
        except KeyError:
            f.write('-Metos3DTracerInitValue                 {} \n'.format(opt['/metos3d/initial_constant_concentrations_string']))

        f.write('-Metos3DTracerOutputDirectory           {} \n'.format(opt['/metos3d/tracer_output_dir']))
        f.write('-Metos3DTracerOutputFile                {} \n\n'.format(','.join(map(str, opt['/metos3d/tracer_output_filenames']))))

        f.write('# bgc parameter \n')
        f.write('-Metos3DParameterCount                  {:d} \n'.format(len(opt['/model/parameters'])))
        f.write('-Metos3DParameterValue                  {} \n\n'.format(opt['/metos3d/parameters_string']))

        f.write('# bgc boundary conditions \n')
        f.write('-Metos3DBoundaryConditionCount          2 \n')
        f.write('-Metos3DBoundaryConditionInputDirectory {}/Forcing/BoundaryCondition/ \n'.format(opt['/metos3d/data_dir']))
        f.write('-Metos3DBoundaryConditionName           Latitude,IceCover \n')
        f.write('-Metos3DLatitudeCount                   1 \n')
        f.write('-Metos3DLatitudeFileFormat              latitude.petsc \n')
        f.write('-Metos3DIceCoverCount                   12 \n')
        f.write('-Metos3DIceCoverFileFormat              fice_$02d.petsc \n\n')

        f.write('# bgc domain conditions \n')
        f.write('-Metos3DDomainConditionCount            2 \n')
        f.write('-Metos3DDomainConditionInputDirectory   {}/Forcing/DomainCondition/ \n'.format(opt['/metos3d/data_dir']))
        f.write('-Metos3DDomainConditionName             LayerDepth,LayerHeight \n')
        f.write('-Metos3DLayerDepthCount                 1 \n')
        f.write('-Metos3DLayerDepthFileFormat            z.petsc \n\n')
        f.write('-Metos3DLayerHeightCount                1 \n')
        f.write('-Metos3DLayerHeightFileFormat           dz.petsc \n')

        f.write('# transport \n')
        f.write('-Metos3DTransportType                   Matrix \n')
        f.write('-Metos3DMatrixInputDirectory            {}/Transport/Matrix5_4/{:d}dt/ \n'.format(opt['/metos3d/data_dir'], opt['/model/time_step_multiplier']))
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
            f.write('-Metos3DSpinupTolerance                 {:f} \n'.format(opt['/model/spinup/tolerance']))
        except KeyError:
            pass
        f.write('-Metos3DSpinupCount                     {:d} \n'.format(opt['/model/spinup/years']))

        if opt['/metos3d/write_trajectory']:
            f.write('-Metos3DSpinupMonitorFileFormatPrefix   sp$0004d-,ts$0004d- \n')
            f.write('-Metos3DSpinupMonitorModuloStep         1,1 \n')

        util.io.fs.flush_and_close(f)


        ## write job file
        run_command = '{} {} \n'.format(opt['/metos3d/sim_file'], opt['/metos3d/option_file'])
        super().write_job_file(run_command, modules=['intel16', 'intelmpi16'])

        logger.debug('Job initialised.')

    
    
    ## check integrity

    def check_integrity(self, should_be_started=False, should_be_readonly=False):
        ## super check
        super().check_integrity(should_be_started=should_be_started, should_be_readonly=should_be_readonly)
        
        ## check output
        if self.is_started():
            self.time_step
        
        if self.is_finished():
            self.last_year
            self.last_tolerance
        
        ## check functions
        options = self.options
        
        def option_exists(option):
            try:
                options[option]
            except KeyError:
                return False
            else:
                return True
        
        def check_if_option_exists(option, should_exists=True):
            exists = option_exists(option)
            if not exists and should_exists:
                raise util.batch.universal.system.JobMissingOptionError(self, option)
            if exists and not should_exists:
                raise util.batch.universal.system.JobError(self, 'Job option {} should not exist!'.format(option))

        def check_if_file_exists(file, should_exists=True, should_be_in_output_dir=True):
            if should_be_in_output_dir and not file.startswith(self.output_dir):
                raise util.batch.universal.system.JobError(self, 'The file {} should start with {}.'.format(file, self.output_dir))
            exists =  os.path.exists(file)
            if should_exists and not exists:
                raise util.batch.universal.system.JobError(self, 'File {} does not exist.'.format(file))
            if not should_exists and exists:
                raise util.batch.universal.system.JobError(self, 'File {} should not exist.'.format(file))
    
        def check_if_file_option_exists(option, should_exists=True, check_file_exists=True, should_be_in_output_dir=True):
            check_if_option_exists(option, should_exists=should_exists)
            try:
                file = options[option]
            except KeyError:
                pass
            else:
                check_if_file_exists(file, should_exists=should_exists, should_be_in_output_dir=should_be_in_output_dir)
        
        ## options should always exist
        for option in ['/model/name', '/model/tracer', '/model/time_step', '/model/time_steps_per_year', '/model/time_step_multiplier', '/model/spinup/years', '/model/parameters', '/metos3d/parameters_string', '/metos3d/debuglevel', '/metos3d/write_trajectory', '/metos3d/tracer_output_filenames']:
            check_if_option_exists(option)
        
        # for option in ['/metos3d/data_dir', '/metos3d/sim_file']:
        #     check_if_file_option_exists(option, should_be_in_output_dir=False)
        
        for option in ['/metos3d/tracer_output_dir', '/metos3d/output_dir', '/metos3d/option_file']:
            check_if_file_option_exists(option, should_be_in_output_dir=True)
    
        
        ## tracer input files
        tracer_input_options = ['/model/tracer_input_files', '/metos3d/tracer_input_dir', '/metos3d/tracer_input_filenames']
        tracer_input_options_exist = tuple(map(option_exists, tracer_input_options))
        
        if not all(tracer_input_options_exist) and not all(map(lambda t: not t, tracer_input_options_exist)):
            raise util.batch.universal.system.JobError(self, 'Some of the options {} exist and some of them not!'.format(tracer_input_options))
        
        tracer_input_use = all(tracer_input_options_exist)
        if tracer_input_use:
            tuple(map(lambda file: check_if_file_exists(file, should_be_in_output_dir=False), options['/model/tracer_input_files']))
            tuple(map(lambda file: check_if_file_exists(file, should_be_in_output_dir=True), [os.path.join(options['/metos3d/tracer_input_dir'], filename) for filename in options['/metos3d/tracer_input_filenames']]))
        
        ## concentrations
        check_if_option_exists('/model/initial_constant_concentrations', should_exists=not tracer_input_use)
        check_if_option_exists('/metos3d/initial_constant_concentrations_string', should_exists=not tracer_input_use)
        
        ## tracer dirs
        if options['/metos3d/output_dir'] != options['/metos3d/tracer_output_dir']:
            raise util.batch.universal.system.JobError(self, 'Metos3D output dir {} and tracer output dir in job file in {} are not the same.'.format(options['/metos3d/output_dir'], options['/metos3d/tracer_output_dir']))
        
        if tracer_input_use:
            if options['/metos3d/tracer_input_dir'] != options['/metos3d/tracer_output_dir']:
                raise util.batch.universal.system.JobError(self, 'Metos3D tracer input dir {} and tracer output dir in job file in {} are not the same.'.format(options['/metos3d/tracer_input_dir'], options['/metos3d/tracer_output_dir']))
        
        ## tracer output files
        tracer_output_files = tuple(map(lambda filename: os.path.join(options['/metos3d/tracer_output_dir'], filename), options['/metos3d/tracer_output_filenames']))
        tuple(map(lambda file: check_if_file_exists(file, should_exists=not self.is_running(), should_be_in_output_dir=True), tracer_output_files))
