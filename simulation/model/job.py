import os
import time
import re
import warnings

import numpy as np

import simulation.model.constants

import util.batch.universal.system
import util.io.fs
import util.petsc.universal

import util.logging


class Metos3D_Job(util.batch.universal.system.Job):

    IGNORE_ERROR_KEYWORDS = tuple(error_message.lower() for error_message in (
        "Error_Path = ",
        "cpuinfo: error while loading shared libraries: libgcc_s.so.1: cannot open shared object file: No such file or directory",
        "<class 'socket.error'>",
        "failed to create a socket (sock2): <class 'socket.error'>, [Errno 98] Address already in use",
        "failed to connect to the socket (sock2): {<type 'exceptions.UnboundLocalError'>, local variable 'sock2' referenced before assignment}.",
        "UnboundLocalError: local variable 'sock2' referenced before assignment"
    ))

    # run options

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

    @property
    def model_parameters(self):
        return self.options['/model/parameters']

    # files

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

    def remove_tracer_info_files(self, force=False, not_exist_okay=False):
        for file in self.tracer_input_info_files + self.tracer_output_info_files:
            util.io.fs.remove_file(file, force=force, not_exist_okay=not_exist_okay)

    # exit code and is finished

    @property
    def exit_code(self):
        # get metos3d exit code
        exit_code = super().exit_code
        if exit_code != 0:
            return exit_code

        # check if output file exists
        if self.output_file is not None and not os.path.exists(self.output_file):
            raise util.batch.universal.system.JobError(self, 'Output file is missing!')

        # check output file for errors
        self.check_output_file()

        # everything is okay
        return 0

    def is_finished(self, check_exit_code=True):
        # check if finished without exit code check
        if not super().is_finished(check_exit_code=False):
            return False

        # ensure that output file exists for error check
        if check_exit_code and self.output_file is not None and not os.path.exists(self.output_file):
            return False

        # check if finished with exit code check
        if check_exit_code and not super().is_finished(check_exit_code=check_exit_code):
            return False

        # check if output file is completely written
        job_output = self.output
        if 'Metos3DFinal' in job_output:
            return True
        else:
            time.sleep(30)
            if self.output != job_output:
                return False
            else:
                raise util.batch.universal.system.JobError(self, 'The job output file is not completely written!', job_output)

    # write job file

    def write_job_file(self, model_name, model_parameters, years, tolerance=None, time_step=1, initial_constant_concentrations=None, tracer_input_files=None, total_concentration_factor=1, write_trajectory=False, job_options=None):

        util.logging.debug('Initialising job with model {}, parameters {},  years {}, tolerance {}, time step {}, initial_constant_concentrations {}, tracer_input_files {}, total concentration factor {} and job_options {}.'.format(model_name, model_parameters, years, tolerance, time_step, initial_constant_concentrations, tracer_input_files, total_concentration_factor, job_options))

        # check input
        if time_step not in simulation.model.constants.METOS_TIME_STEPS:
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

        # unpack job setup
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

        # prepare job name
        if len(job_name) > 0:
            job_name += '_'
        job_name += '{}_{}_{}'.format(model_name, years, time_step)

        # use best node setup if no node setup passed
        if nodes_setup is None:
            nodes_setup = util.batch.universal.system.NodeSetup()

        # check/set memory
        if nodes_setup.memory is None:
            nodes_setup.memory = simulation.model.constants.JOB_MEMORY_GB
        elif nodes_setup.memory < simulation.model.constants.JOB_MEMORY_GB:
            util.logging.warn('The chosen memory {} is below the needed memory {}. Changing to needed memory.'.format(nodes_setup.memory, simulation.model.constants.JOB_MEMORY_GB))
            nodes_setup.memory = simulation.model.constants.JOB_MEMORY_GB

        # check/set walltime
        sec_per_year = 10 * np.exp(- (nodes_setup.nodes * nodes_setup.cpus) / 80) + 3
        sec_per_year /= time_step**(0.5)
        estimated_walltime_hours = np.ceil(years * sec_per_year / 60**2)
        util.logging.debug('The estimated walltime for {} nodes with {} cpus, {} years and time step {} is {} hours.'.format(nodes_setup.nodes, nodes_setup.cpus, years, time_step, estimated_walltime_hours))
        if nodes_setup.walltime is None:
            nodes_setup.walltime = estimated_walltime_hours
        else:
            if nodes_setup.walltime < estimated_walltime_hours:
                util.logging.debug('The chosen walltime {} for the job with {} years, {} nodes and {} cpus is below the estimated walltime {}.'.format(nodes_setup.walltime, years, nodes_setup.nodes, nodes_setup.cpus, estimated_walltime_hours))

        # check/set min cpus
        if nodes_setup.total_cpus_min is None:
            nodes_setup.total_cpus_min = min(int(np.ceil(years / 20)), 32)

        # check/set max nodes
        if nodes_setup.nodes_max is None and years <= 1:
            nodes_setup.nodes_max = 1

        # init job
        super().set_job_options(job_name, nodes_setup)

        # get output dir
        output_dir = self.output_dir
        output_dir_not_expanded = os.path.join(self.output_dir_not_expanded, '')  # ending with separator

        # set model options
        opt = self.options

        opt['/model/name'] = model_name
        opt['/model/tracers'] = simulation.model.constants.MODEL_TRACER[model_name]

        time_steps_per_year = simulation.model.constants.METOS_T_DIM / time_step
        assert time_steps_per_year.is_integer()
        time_steps_per_year = int(time_steps_per_year)
        opt['/model/time_step'] = 1 / time_steps_per_year
        opt['/model/time_steps_per_year'] = time_steps_per_year
        opt['/model/time_step_multiplier'] = time_step

        opt['/model/spinup/years'] = years
        if tolerance is not None:
            opt['/model/spinup/tolerance'] = tolerance

        # set metos3d files and dirs
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
        opt['/metos3d/tracer_output_filenames'] = ['{}_output.petsc'.format(tracer) for tracer in opt['/model/tracers']]

        # tracer_input_files
        if tracer_input_files is not None:
            opt['/model/tracer_input_files'] = tracer_input_files

            opt['/metos3d/tracer_input_dir'] = output_dir_not_expanded
            opt['/metos3d/tracer_input_filenames'] = ['{}_input.petsc'.format(tracer) for tracer in opt['/model/tracers']]

            for i in range(len(opt['/model/tracers'])):
                # check existence of input files
                tracer_input_file_base = os.path.expanduser(os.path.expandvars(tracer_input_files[i]))
                if not os.path.exists(tracer_input_file_base):
                    raise FileNotFoundError(tracer_input_file_base)
                # make input files for metos3d
                tracer_input_file_metos3d = os.path.join(output_dir, opt['/metos3d/tracer_input_filenames'][i])
                if total_concentration_factor == 1:
                    tracer_input_file_base_rel = os.path.relpath(tracer_input_file_base, start=output_dir)
                    os.symlink(tracer_input_file_base_rel, tracer_input_file_metos3d)
                    # sometimes relative links does not work on file systems so absolute links are a fallback
                    if not os.path.exists(tracer_input_file_metos3d):
                        warnings.warn(f'Relative symbolic link from {output_dir} to {tracer_input_file_base_rel} does not work. Trying absolute symbolic link.')
                        tracer_input_file_base_abs = os.path.abspath(tracer_input_file_base)
                        os.remove(tracer_input_file_metos3d)
                        os.symlink(tracer_input_file_base_abs, tracer_input_file_metos3d)
                else:
                    tracer_input = util.petsc.universal.load_petsc_vec_to_numpy_array(tracer_input_file_base)
                    tracer_input = tracer_input * total_concentration_factor
                    util.petsc.universal.save_numpy_array_to_petsc_vec(tracer_input_file_metos3d, tracer_input)
                assert os.path.exists(tracer_input_file_metos3d)

        # initial_constant_concentrations
        else:
            if initial_constant_concentrations is None:
                initial_constant_concentrations = simulation.model.constants.MODEL_DEFAULT_INITIAL_CONCENTRATION[model_name]
            initial_constant_concentrations = initial_constant_concentrations * total_concentration_factor
            opt['/model/initial_constant_concentrations'] = initial_constant_concentrations

            initial_constant_concentrations_string = ','.join(map(str, initial_constant_concentrations))
            opt['/metos3d/initial_constant_concentrations_string'] = initial_constant_concentrations_string

        # model parameter
        model_parameters = np.asarray(model_parameters, dtype=np.float64)
        opt['/model/parameters'] = model_parameters

        model_parameters_string = ','.join(map(lambda f: simulation.model.constants.DATABASE_PARAMETERS_FORMAT_STRING.format(f), model_parameters))
        opt['/metos3d/parameters_string'] = model_parameters_string

        # prepare metos3d options
        linesep = os.linesep

        metos3d_options = []

        metos3d_options.append('# debug')
        metos3d_options.append('-Metos3DDebugLevel                      {:d}'.format(opt['/metos3d/debuglevel']))
        metos3d_options.append(linesep)

        metos3d_options.append('# geometry')
        metos3d_options.append('-Metos3DGeometryType                    Profile')
        metos3d_options.append('-Metos3DProfileInputDirectory           {}/Geometry/'.format(opt['/metos3d/data_dir']))
        metos3d_options.append('-Metos3DProfileMaskFile                 landSeaMask.petsc')
        metos3d_options.append('-Metos3DProfileVolumeFile               volumes.petsc')
        metos3d_options.append(linesep)

        metos3d_options.append('# bgc tracer')
        metos3d_options.append('-Metos3DTracerCount                     {:d}'.format(len(opt['/model/tracers'])))
        try:
            metos3d_options.append('-Metos3DTracerInputDirectory            {}'.format(opt['/metos3d/tracer_input_dir']))
            metos3d_options.append('-Metos3DTracerInitFile                  {}'.format(','.join(map(str, opt['/metos3d/tracer_input_filenames']))))
        except KeyError:
            metos3d_options.append('-Metos3DTracerInitValue                 {}'.format(opt['/metos3d/initial_constant_concentrations_string']))
        metos3d_options.append('-Metos3DTracerOutputDirectory           {}'.format(opt['/metos3d/tracer_output_dir']))
        metos3d_options.append('-Metos3DTracerOutputFile                {}'.format(','.join(map(str, opt['/metos3d/tracer_output_filenames']))))
        metos3d_options.append(linesep)

        metos3d_options.append('# bgc parameter')
        metos3d_options.append('-Metos3DParameterCount                  {:d}'.format(len(opt['/model/parameters'])))
        metos3d_options.append('-Metos3DParameterValue                  {}'.format(opt['/metos3d/parameters_string']))
        metos3d_options.append(linesep)

        metos3d_options.append('# bgc boundary conditions')
        metos3d_options.append('-Metos3DBoundaryConditionCount          2')
        metos3d_options.append('-Metos3DBoundaryConditionInputDirectory {}/Forcing/BoundaryCondition/'.format(opt['/metos3d/data_dir']))
        metos3d_options.append('-Metos3DBoundaryConditionName           Latitude,IceCover')
        metos3d_options.append('-Metos3DLatitudeCount                   1')
        metos3d_options.append('-Metos3DLatitudeFileFormat              latitude.petsc')
        metos3d_options.append('-Metos3DIceCoverCount                   12')
        metos3d_options.append('-Metos3DIceCoverFileFormat              fice_$02d.petsc')
        metos3d_options.append(linesep)

        metos3d_options.append('# bgc domain conditions')
        metos3d_options.append('-Metos3DDomainConditionCount            2')
        metos3d_options.append('-Metos3DDomainConditionInputDirectory   {}/Forcing/DomainCondition/'.format(opt['/metos3d/data_dir']))
        metos3d_options.append('-Metos3DDomainConditionName             LayerDepth,LayerHeight')
        metos3d_options.append('-Metos3DLayerDepthCount                 1')
        metos3d_options.append('-Metos3DLayerDepthFileFormat            z.petsc')
        metos3d_options.append('-Metos3DLayerHeightCount                1')
        metos3d_options.append('-Metos3DLayerHeightFileFormat           dz.petsc')

        metos3d_options.append('# transport')
        metos3d_options.append('-Metos3DTransportType                   Matrix')
        metos3d_options.append('-Metos3DMatrixInputDirectory            {}/Transport/Matrix5_4/{:d}dt/'.format(opt['/metos3d/data_dir'], opt['/model/time_step_multiplier']))
        metos3d_options.append('-Metos3DMatrixCount                     12')
        metos3d_options.append('-Metos3DMatrixExplicitFileFormat        Ae_$02d.petsc')
        metos3d_options.append('-Metos3DMatrixImplicitFileFormat        Ai_$02d.petsc')
        metos3d_options.append(linesep)

        metos3d_options.append('# time stepping')
        metos3d_options.append('-Metos3DTimeStepStart                   0.0')
        metos3d_options.append('-Metos3DTimeStepCount                   {:d}'.format(opt['/model/time_steps_per_year']))
        metos3d_options.append('-Metos3DTimeStep                        {:.18f}'.format(opt['/model/time_step']))
        metos3d_options.append(linesep)

        metos3d_options.append('# solver')
        metos3d_options.append('-Metos3DSolverType                      Spinup')
        metos3d_options.append('-Metos3DSpinupMonitor')
        try:
            metos3d_options.append('-Metos3DSpinupTolerance                 {:f}'.format(opt['/model/spinup/tolerance']))
        except KeyError:
            pass
        metos3d_options.append('-Metos3DSpinupCount                     {:d}'.format(opt['/model/spinup/years']))

        if opt['/metos3d/write_trajectory']:
            metos3d_options.append('-Metos3DSpinupMonitorFileFormatPrefix   sp$0004d-,ts$0004d-')
            metos3d_options.append('-Metos3DSpinupMonitorModuloStep         1,1')
        metos3d_options.append(linesep)

        metos3d_options = linesep.join(metos3d_options)

        # write metos3d option file
        with open(opt['/metos3d/option_file'], mode='w') as f:
            f.write(metos3d_options)
            f.flush()
            os.fsync(f.fileno())

        # write job file
        batch_system = util.batch.universal.system.BATCH_SYSTEM
        pre_command = batch_system.pre_command('metos3d')
        pre_command += linesep + 'export OMP_NUM_THREADS=1' + linesep
        command = '{} {}'.format(opt['/metos3d/sim_file'], opt['/metos3d/option_file']) + linesep
        super().write_job_file(command, pre_command=pre_command, use_mpi=True, use_conda=False, add_timing=True)

        util.logging.debug('Job initialised.')

    # check integrity

    def check_integrity(self, force_to_be_started=False, force_to_be_readonly=False):
        # super check
        super().check_integrity(force_to_be_started=force_to_be_started, force_to_be_readonly=force_to_be_readonly)

        # check output
        if self.is_started():
            self.time_step

        if self.is_finished():
            self.last_year
            self.last_tolerance

        # check functions
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
            exists = os.path.exists(file)
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

        # options should always exist
        for option in ['/model/name', '/model/tracers', '/model/time_step', '/model/time_steps_per_year', '/model/time_step_multiplier', '/model/spinup/years', '/model/parameters', '/metos3d/parameters_string', '/metos3d/debuglevel', '/metos3d/write_trajectory', '/metos3d/tracer_output_filenames']:
            check_if_option_exists(option)

        # for option in ['/metos3d/data_dir', '/metos3d/sim_file']:
        #     check_if_file_option_exists(option, should_be_in_output_dir=False)

        for option in ['/metos3d/tracer_output_dir', '/metos3d/output_dir', '/metos3d/option_file']:
            check_if_file_option_exists(option, should_be_in_output_dir=True)

        # tracer input files
        tracer_input_options = ['/model/tracer_input_files', '/metos3d/tracer_input_dir', '/metos3d/tracer_input_filenames']
        tracer_input_options_exist = tuple(map(option_exists, tracer_input_options))

        if not all(tracer_input_options_exist) and not all(map(lambda t: not t, tracer_input_options_exist)):
            raise util.batch.universal.system.JobError(self, 'Some of the options {} exist and some of them not!'.format(tracer_input_options))

        tracer_input_use = all(tracer_input_options_exist)
        if tracer_input_use:
            tuple(map(lambda file: check_if_file_exists(file, should_be_in_output_dir=False), options['/model/tracer_input_files']))
            tuple(map(lambda file: check_if_file_exists(file, should_be_in_output_dir=True), [os.path.join(options['/metos3d/tracer_input_dir'], filename) for filename in options['/metos3d/tracer_input_filenames']]))

        # concentrations
        check_if_option_exists('/model/initial_constant_concentrations', should_exists=not tracer_input_use)
        check_if_option_exists('/metos3d/initial_constant_concentrations_string', should_exists=not tracer_input_use)

        # tracer dirs
        if options['/metos3d/output_dir'] != options['/metos3d/tracer_output_dir']:
            raise util.batch.universal.system.JobError(self, 'Metos3D output dir {} and tracer output dir in job file in {} are not the same.'.format(options['/metos3d/output_dir'], options['/metos3d/tracer_output_dir']))

        if tracer_input_use:
            if options['/metos3d/tracer_input_dir'] != options['/metos3d/tracer_output_dir']:
                raise util.batch.universal.system.JobError(self, 'Metos3D tracer input dir {} and tracer output dir in job file in {} are not the same.'.format(options['/metos3d/tracer_input_dir'], options['/metos3d/tracer_output_dir']))

        # tracer output files
        tracer_output_files = tuple(map(lambda filename: os.path.join(options['/metos3d/tracer_output_dir'], filename), options['/metos3d/tracer_output_filenames']))
        tuple(map(lambda file: check_if_file_exists(file, should_exists=not self.is_running(), should_be_in_output_dir=True), tracer_output_files))
