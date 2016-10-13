import os
import stat
import tempfile
import warnings

import numpy as np

import util.io.fs
import util.index_database.array_and_txt_file_based
import util.index_database.petsc_file_based
import util.pattern
import util.math.interpolate
import util.math.finite_differences
import util.batch.universal.system
import util.options
import util.cache.memory_based
import util.logging

import measurements.land_sea_mask.lsm
import measurements.universal.data

import simulation.constants
import simulation.model.data
import simulation.model.job
import simulation.model.options
import simulation.model.constants

logger = util.logging.logger



class Model_Database:
    
    def __init__(self, model_options=None, job_options=None):
        logger.debug('Model initiated with model_options {} and job setup {}.'.format(model_options, job_options))
        
        ## set model options
        model_options = util.options.as_options(model_options, simulation.model.options.ModelOptions)
        self.model_options = model_options
        
        self.database_output_dir = simulation.model.constants.DATABASE_OUTPUT_DIR
        self.start_from_closest_parameters = simulation.model.constants.MODEL_START_FROM_CLOSEST_PARAMETER_SET
        self.model_spinup_max_years = simulation.model.constants.MODEL_SPINUP_MAX_YEARS
        self._cached_interpolator = None
        
        self.model_lsm = simulation.model.constants.METOS_LSM
        
        
        ## set job setup collection
        # convert job setup to job setup collection
        if job_options is None:
            job_options = {}

        keys = list(job_options.keys())
        kinds = ['spinup', 'derivative', 'trajectory']
        if not any(kind in keys for kind in kinds):
            job_options = {'spinup': job_options}

        # if not passed, use default job setups
        try:
            job_options['spinup']
        except KeyError:
            job_options['spinup'] = {}
        try:
            job_options['spinup']['name']
        except KeyError:
            job_options['spinup']['name'] = 'spinup'
            default_name = ''
        else:
            default_name = job_options['spinup']['name'] 
            
        try:
            job_options['derivative']
        except KeyError:
            job_options['derivative'] = job_options['spinup'].copy()
            del job_options['derivative']['name']
        try:
            job_options['derivative']['name']
        except KeyError:
            job_options['derivative']['name'] = 'derivative' + default_name
        try:
            job_options['trajectory']
        except KeyError:
            job_options['trajectory'] = {}
            job_options['trajectory']['nodes_setup'] = util.batch.universal.system.NodeSetup(nodes_max=1, memory=simulation.model.constants.JOB_MEMORY_GB)
        try:
            job_options['trajectory']['name']
        except KeyError:
            job_options['trajectory']['name'] = 'trajectory' + default_name

        self.job_options = job_options
    
    
    ## model dir
    
    @property
    def model_dir(self):
        model_name = self.model_options.model_name
        model_dirname = simulation.model.constants.DATABASE_MODEL_DIRNAME.format(model_name)
        model_dir = os.path.join(self.database_output_dir, model_dirname)
        
        logger.debug('Returning model directory {} for model {}.'.format(model_dir, model_name))
        return model_dir
    
    
    ## concentration dir
    
    @property
    def initial_concentration_base_dir(self):
        use_constant_concentrations = self.model_options.initial_concentration_options.use_constant_concentrations
        
        if use_constant_concentrations:
            initial_concentration_base_dirname = simulation.model.constants.DATABASE_CONSTANT_CONCENTRATIONS_DIRNAME
        else:
            initial_concentration_base_dirname = simulation.model.constants.DATABASE_VECTOR_CONCENTRATIONS_DIRNAME
            
        initial_concentration_base_dir = os.path.join(self.model_dir, initial_concentration_base_dirname)
        logger.debug('Returning initial concentration directory {} for use constant concentration {}.'.format(initial_concentration_base_dir, use_constant_concentrations))
        return initial_concentration_base_dir
    
    
    @property
    def _constant_concentrations_db(self):
        model_dir = self.model_dir
        tolerance_options = self.model_options.initial_concentration_options.tolerance_options

        value_file = os.path.join(model_dir, simulation.model.constants.DATABASE_CONSTANT_CONCENTRATIONS_DIRNAME, simulation.model.constants.DATABASE_CONCENTRATIONS_DIRNAME, simulation.model.constants.DATABASE_CONSTANT_CONCENTRATIONS_FILENAME)
        array_file = os.path.join(model_dir, simulation.model.constants.DATABASE_CONSTANT_CONCENTRATIONS_DIRNAME, simulation.model.constants.DATABASE_CONSTANT_CONCENTRATIONS_LOOKUP_ARRAY_FILENAME)
        
        constant_concentrations_db = util.index_database.array_and_txt_file_based.Database(array_file, value_file, value_reliable_decimal_places=simulation.model.constants.DATABASE_CONSTANT_CONCENTRATIONS_RELIABLE_DECIMAL_PLACES, tolerance_options=tolerance_options)
        return constant_concentrations_db
    
    
    @property
    def _vector_concentrations_db(self):
        model_dir = self.model_dir
        tracers = self.model_options.tracers
        tolerance_options = self.model_options.initial_concentration_options.tolerance_options
        
        value_dir = os.path.join(model_dir, simulation.model.constants.DATABASE_VECTOR_CONCENTRATIONS_DIRNAME, simulation.model.constants.DATABASE_CONCENTRATIONS_DIRNAME)
        concentration_filenames = [simulation.model.constants.DATABASE_VECTOR_CONCENTRATIONS_FILENAME.format(tracer=tracer) for tracer in tracers]
        
        vector_concentrations_db = util.index_database.petsc_file_based.Database(value_dir, concentration_filenames, value_reliable_decimal_places=simulation.model.constants.DATABASE_VECTOR_CONCENTRATIONS_RELIABLE_DECIMAL_PLACES, tolerance_options=tolerance_options)
        return vector_concentrations_db

    
    @property
    def initial_concentration_dir_index(self):
        initial_concentration_options = self.model_options.initial_concentration_options
        
        ## search for directories with matching concentration
        logger.debug('Searching concentration directory for concentration {} .'.format(initial_concentration_options))

        concentrations = initial_concentration_options.concentrations
        if initial_concentration_options.use_constant_concentrations:
            concentration_db = self._constant_concentrations_db
        else:
            concentration_db = self._vector_concentrations_db
        
        index = concentration_db.get_or_add_index(concentrations)
        assert index is not None
        return index
    
    
    def initial_concentration_dir_with_index(self, index):
        if index is not None:
            dir = os.path.join(self.initial_concentration_base_dir, simulation.model.constants.DATABASE_CONCENTRATIONS_DIRNAME.format(index))
            logger.debug('Returning initial concentration directory {} for index {}.'.format(dir, index))
            return dir
        else:
            return None
    
    
    @property
    def initial_concentration_dir(self):
        index = self.initial_concentration_dir_index
        concentration_set_dir = self.initial_concentration_dir_with_index(index)

        logger.debug('Matching directory for concentrations found at {}.'.format(concentration_set_dir))
        assert concentration_set_dir is not None
        return concentration_set_dir
    
    
    @property
    def initial_concentration_files(self):
        assert not self.model_options.initial_concentration_options.use_constant_concentrations
        index = self.initial_concentration_dir_index
        concentration_db = self._vector_concentrations_db
        concentration_files = concentration_db.value_files(index)

        logger.debug('Using concentration files {}.'.format(concentration_files))
        assert concentration_files is not None
        return concentration_files

    
    ## time step dir
    
    @property
    def time_step_dir(self):
        time_step = self.model_options.time_step
        initial_concentration_dir = self.initial_concentration_dir
        time_step_dirname = simulation.model.constants.DATABASE_TIME_STEP_DIRNAME.format(time_step)
        time_step_dir = os.path.join(initial_concentration_dir, time_step_dirname, '')
        logger.debug('Returning time step directory {} for time step {}.'.format(time_step_dir, time_step))
        return time_step_dir
    
    
    ## parameter set dir
    
    def parameter_set_dir_with_index(self, index):
        if index is not None:
            dir = os.path.join(self.time_step_dir, simulation.model.constants.DATABASE_PARAMETERS_DIRNAME.format(index))
            logger.debug('Returning parameter set directory {} for index {}.'.format(dir, index))
            return dir
        else:
            return None
    
    
    @property
    def _parameter_db(self):
        time_step_dir = self.time_step_dir
        parameter_tolerance_options = self.model_options.parameter_tolerance_options
        
        array_file = os.path.join(time_step_dir, simulation.model.constants.DATABASE_PARAMETERS_LOOKUP_ARRAY_FILENAME)
        value_file = os.path.join(time_step_dir, simulation.model.constants.DATABASE_PARAMETERS_DIRNAME, simulation.model.constants.DATABASE_PARAMETERS_FILENAME)
        
        parameter_db = util.index_database.array_and_txt_file_based.Database(array_file, value_file, value_reliable_decimal_places=simulation.model.constants.DATABASE_PARAMETERS_RELIABLE_DECIMAL_PLACES, tolerance_options=parameter_tolerance_options)
        return parameter_db


    @property
    def parameter_set_dir(self):
        ## search for directories with matching parameters
        parameters = self.model_options.parameters
        logger.debug('Searching parameter directory for parameters {}.'.format(parameters))
        
        index = self._parameter_db.get_or_add_index(parameters)
        parameter_set_dir = self.parameter_set_dir_with_index(index)
        
        ## return
        logger.debug('Matching directory for parameters found at {}.'.format(parameter_set_dir))
        assert parameter_set_dir is not None
        return parameter_set_dir
    
    
    @property
    def closest_parameter_set_dir(self):
        parameters = self.model_options.parameters
        logger.debug('Searching for directory for parameters as close as possible to {}.'.format(parameters))
        
        ## get closest indices
        closest_indices = self._parameter_db.closest_indices(parameters)
        
        ## check if run dirs exist
        i = 0
        while i < len(closest_indices) and self.last_run_dir(self.spinup_dir_with_index[i]) is None:
            i = i +1
        if i < len(closest_indices):
            closest_index = closest_indices[i]
        else:
            closest_index = None
        
        ## get parameter set dir and return 
        closest_parameter_set_dir = self.parameter_set_dir_with_index(closest_index)
        logger.debug('Closest parameter set dir is {}.'.format(closest_parameter_set_dir))
        return closest_parameter_set_dir


    ## spinup dir

    def spinup_dir_with_index(self, index):
        if index is not None:
            dir = os.path.join(self.parameter_set_dir_with_index(index), simulation.model.constants.DATABASE_SPINUP_DIRNAME)
            logger.debug('Returning spinup directory {} for index {}.'.format(dir, index))
            return dir
        else:
            return None

    
    @property
    def spinup_dir(self):
        spinup_dir = os.path.join(self.parameter_set_dir, simulation.model.constants.DATABASE_SPINUP_DIRNAME)
        logger.debug('Returning spinup directory {}.'.format(spinup_dir))
        return spinup_dir
    
    
    @property
    def closest_spinup_dir(self):
        spinup_dir = os.path.join(self.closest_parameter_set_dir, simulation.model.constants.DATABASE_SPINUP_DIRNAME)
        logger.debug('Returning closest spinup directory {}.'.format(spinup_dir))
        return spinup_dir
    
    
    ## run dirs
    
    @property
    def run_dir(self):
        spinup_options = self.model_options.spinup_options
        run_dir = self.matching_run_dir(spinup_options)
        return run_dir
    

    def run_dirs(self, search_path):
        DATABASE_RUN_DIRNAME_REGULAR_EXPRESSION = util.pattern.convert_format_string_in_regular_expression(simulation.model.constants.DATABASE_RUN_DIRNAME)
        try:
            run_dirs = util.io.fs.find_with_regular_expression(search_path, DATABASE_RUN_DIRNAME_REGULAR_EXPRESSION, exclude_files=True, use_absolute_filenames=False, recursive=False)
        except OSError as exception:
            logger.warn('It could not been searched in the search path "{}": {}'.format(search_path, exception))
            run_dirs = []

        return run_dirs


    def last_run_dir(self, search_path):
        logger.debug('Searching for last run in {}.'.format(search_path))

        last_run_index =  len(self.run_dirs(search_path)) - 1
        
        if last_run_index >= 0:
            last_run_dirname = simulation.model.constants.DATABASE_RUN_DIRNAME.format(last_run_index)
            last_run_dir = os.path.join(search_path, last_run_dirname)
            
            ## check job options file
            with simulation.model.job.Metos3D_Job(last_run_dir, force_load=True) as job:
                pass
        else:
            last_run_dir = None

        logger.debug('Returning last run directory {}.'.format(last_run_dir))
        return last_run_dir


    def previous_run_dir(self, run_dir):
        (spinup_dir, run_dirname) = os.path.split(run_dir)
        run_index = util.pattern.get_int_in_string(run_dirname)
        if run_index > 0:
            previous_run_dirname = simulation.model.constants.DATABASE_RUN_DIRNAME.format(run_index - 1)
            previous_run_dir = os.path.join(spinup_dir, previous_run_dirname)
        else:
            previous_run_dir = None

        return previous_run_dir
    
    
    def make_new_run_dir(self, output_path):
        ## get next run index
        os.makedirs(output_path, exist_ok=True)
        next_run_index = len(self.run_dirs(output_path))

        ## create run dir
        run_dirname = simulation.model.constants.DATABASE_RUN_DIRNAME.format(next_run_index)
        run_dir = os.path.join(output_path, run_dirname)

        logger.debug('Creating new run directory {} at {}.'.format(run_dir, output_path))
        os.makedirs(run_dir, exist_ok=False)
        return run_dir
    
    
    def matching_run_dir(self, spinup_options):
        spinup_options = util.options.as_options(spinup_options, simulation.model.options.SpinupOptions)
        
        ## get spinup dir
        spinup_dir = self.spinup_dir
        logger.debug('Searching for matching spinup run with options {} in {}.'.format(spinup_options, spinup_dir))

        ## get last run dir
        last_run_dir = self.last_run_dir(spinup_dir)

        ## matching run found
        if self.is_run_matching_options(last_run_dir, spinup_options):
            run_dir = last_run_dir
            logger.debug('Matching spinup run found at {}.'.format(last_run_dir))

        ## create new run
        else:
            logger.debug('No matching spinup run found.')

            ## no previous run exists and starting from closest parameters get last run from closest parameters
            if last_run_dir is None and self.start_from_closest_parameters:
                closest_spinup_dir = self.closest_spinup_dir
                last_run_dir = self.last_run_dir(closest_spinup_dir)

            ## finish last run
            if last_run_dir is not None:
                self.wait_until_run_finished(last_run_dir)

            ## make new run
            years = spinup_options.years
            tolerance = spinup_options.tolerance
            combination = spinup_options.combination
            
            if combination == 'or':
                ## create new run
                run_dir = self.make_new_run_dir(spinup_dir)
                
                ## calculate last years
                if last_run_dir is not None:
                    last_years = self.get_total_years(last_run_dir)
                    logger.debug('Found previous run(s) with total {} years.'.format(last_years))
                else:
                    last_years = 0
                
                ## start new run
                parameters = self.model_options.parameters
                years = years - last_years
                
                initial_concentration_options = self.model_options.initial_concentration_options
                
                if last_run_dir is None:
                    if initial_concentration_options.use_constant_concentrations:
                        constant_concentrations = initial_concentration_options.concentrations
                        self.start_run(parameters, run_dir, years, tolerance=tolerance, job_options=self.job_options_for_kind('spinup'), initial_constant_concentrations=constant_concentrations, wait_until_finished=True)
                    else:
                        concentration_files = self.initial_concentration_files
                        self.start_run(parameters, run_dir, years, tolerance=tolerance, job_options=self.job_options_for_kind('spinup'), tracer_input_files=concentration_files, wait_until_finished=True)
                else:
                    tracer_input_filenames = ['{}_output.petsc'.format(tracer) for tracer in self.model_options.tracers]
                    tracer_input_files = [os.path.join(last_run_dir, tracer_input_file) for tracer_input_file in tracer_input_filenames]
                    self.start_run(parameters, run_dir, years, tolerance=tolerance, job_options=self.job_options_for_kind('spinup'), tracer_input_files=tracer_input_files, wait_until_finished=True)
                    
                
            elif combination == 'and':
                spinup_options = simulation.model.options.SpinupOptions({'years':years, 'tolerance':0, 'combination':'or'})
                run_dir = self.matching_run_dir(spinup_options)
                spinup_options = simulation.model.options.SpinupOptions({'years':self.model_spinup_max_years, 'tolerance':tolerance, 'combination':'or'})
                run_dir = self.matching_run_dir(spinup_options)

            logger.debug('Spinup run directory created at {}.'.format(run_dir))

        return run_dir
    
    
    def start_run(self, model_parameters, output_path, years, tolerance=0, job_options=None, write_trajectory=False, initial_constant_concentrations=None, tracer_input_files=None, total_concentration_factor=1, make_read_only=True, wait_until_finished=True):
        
        model_name = self.model_options.model_name
        time_step = self.model_options.time_step

        ## execute job
        output_path_with_env = output_path.replace(simulation.constants.SIMULATION_OUTPUT_DIR, '${{{}}}'.format(simulation.constants.SIMULATION_OUTPUT_DIR_ENV_NAME))
        with simulation.model.job.Metos3D_Job(output_path_with_env) as job:
            job.write_job_file(model_name, model_parameters, years=years, tolerance=tolerance, time_step=time_step, initial_constant_concentrations=initial_constant_concentrations, tracer_input_files=tracer_input_files, total_concentration_factor=total_concentration_factor, write_trajectory=write_trajectory, job_options=job_options)
            job.start()
            job.make_read_only_input(make_read_only)

        ## wait to finish
        if wait_until_finished:
            self.wait_until_run_finished(output_path, make_read_only=make_read_only)
        else:
            logger.debug('Not waiting for job to finish.')


    ##  access run properties
    
    def wait_until_run_finished(self, run_dir, make_read_only=True):
        with simulation.model.job.Metos3D_Job(run_dir, force_load=True) as job:
            job.make_read_only_input(make_read_only)
            job.wait_until_finished()
            job.make_read_only_output(make_read_only)
    
    def is_run_matching_options(self, run_dir, spinup_options):
        model_spinup_max_years = self.model_spinup_max_years
        spinup_options = util.options.as_options(spinup_options, simulation.model.options.SpinupOptions)

        years = spinup_options.years
        tolerance = spinup_options.tolerance
        combination = spinup_options.combination

        if run_dir is not None:
            run_years = self.get_total_years(run_dir)
            run_tolerance = self.get_real_tolerance(run_dir)

            if combination == 'and':
                is_matching = (run_years >= years and run_tolerance <= tolerance) or run_years >= model_spinup_max_years
                if is_matching and run_tolerance > tolerance:
                    warnings.warn('The run {} does not match the desired tolerance {}, but the max spinup years {} are reached.'.format(run_dir, tolerance, model_spinup_max_years))
            elif combination == 'or':
                is_matching = (run_years >= years or run_tolerance <= tolerance)
            else:
                raise ValueError('Combination "{}" unknown.'.format(combination))
                
            if is_matching:
                logger.debug('Run in {} with years {} and tolerance {} is matching spinup options {}.'.format(run_dir, run_years, run_tolerance, spinup_options))
            else:
                logger.debug('Run in {} with years {} and tolerance {} is not matching spinup options {}.'.format(run_dir, run_years, run_tolerance, spinup_options))
        else:
            is_matching = False
            logger.debug('Run in {} is not matching spinup options {}. No run available.'.format(run_dir, spinup_options))

        return is_matching


    def get_total_years(self, run_dir):
        total_years = 0
        while run_dir is not None:
            with simulation.model.job.Metos3D_Job(run_dir, force_load=True) as job:
                years = job.last_year
            total_years += years
            run_dir = self.previous_run_dir(run_dir)
        return total_years


    def get_real_tolerance(self, run_dir):
        with simulation.model.job.Metos3D_Job(run_dir, force_load=True) as job:
            tolerance = job.last_tolerance
        return tolerance


    def get_time_step(self, run_dir):
        with simulation.model.job.Metos3D_Job(run_dir, force_load=True) as job:
            time_step = job.time_step
        return time_step
    
    
    ## job options

    def job_options_for_kind(self, kind):
        job_options = self.job_options[kind]
        job_options = job_options.copy()
        try:
            job_options['nodes_setup']
        except KeyError:
            pass
        else:
            if job_options['nodes_setup'] is not None:
                job_options['nodes_setup'] = job_options['nodes_setup'].copy()
        return job_options
    
    
    ## iterator
    def iterator(self, model_names=None):
        if model_names is None:
            model_names = simulation.model.constants.MODEL_NAMES
        time_steps = simulation.model.constants.METOS_TIME_STEPS
        old_model_options = self.model_options.copy()
        model_options = self.model_options
        model_options.spinup_options = {'years':1, 'tolerance':0.0, 'combination':'or'}
        
        for model_name in model_names:
            model_options.model_name = model_name
            model_dir = self.model_dir
            if os.path.exists(model_dir):
                concentration_dbs = []
                if os.path.exists(os.path.join(model_dir, simulation.model.constants.DATABASE_CONSTANT_CONCENTRATIONS_DIRNAME)):
                    concentration_dbs.append(self._constant_concentrations_db)
                if os.path.exists(os.path.join(model_dir, simulation.model.constants.DATABASE_VECTOR_CONCENTRATIONS_DIRNAME)):
                    concentration_dbs.append(self._vector_concentrations_db)
                for concentrations_db in concentration_dbs:
                    for concentration in concentrations_db.all_values():
                        model_options.initial_concentration_options.concentrations = concentration
                        for time_step in time_steps:
                            model_options.time_step = time_step
                            if os.path.exists(self.time_step_dir):
                                for parameters in self._parameter_db.all_values():
                                    model_options.parameters = parameters
                                    yield model_options
        
        self.model_options = old_model_options

    ## integrity
    def check_integrity(self, model_names=None):
        logger.debug('Checking database integrity.')
        
        ## check concentrations and parameters database
        if model_names is None:
            model_names = simulation.model.constants.MODEL_NAMES
        
        time_steps = simulation.model.constants.METOS_TIME_STEPS
        
        old_model_options = self.model_options
        model_options = simulation.model.options.ModelOptions()
        self.model_options = model_options
        
        try:
            for model_name in model_names:
                model_options.model_name = model_name
                model_dir = self.model_dir
                if os.path.exists(model_dir):
                    if os.path.exists(os.path.join(model_dir, simulation.model.constants.DATABASE_CONSTANT_CONCENTRATIONS_DIRNAME)):
                        concentrations_db = self._constant_concentrations_db
                        concentrations_db.check_integrity()
                        for concentration in concentrations_db.all_values():
                            model_options.initial_concentration_options.concentrations = concentration
                            for time_step in time_steps:
                                model_options.time_step = time_step
                                if os.path.exists(self.time_step_dir):
                                    parameter_db = self._parameter_db
                                    parameter_db.check_integrity()
        except util.index_database.general.DatabaseError as e:
            logger.error(e)
            raise
        finally:
            self.model_options = old_model_options
        
        ## check that last run dir exists
        for model_option in self.iterator(model_names=model_names):
            spinup_dir = self.spinup_dir
            last_run_dir = self.last_run_dir(spinup_dir)
            if last_run_dir is None:
                raise DatabaseError('It is no run dir in {}!'.format(spinup_dir))



class Model_With_F(Model_Database):
    
    def check_tracers(self, tracers):        
        if tracers is not None:
            tracers = tuple(tracers)
            for tracer in tracers:
                if tracer not in self.model_options.tracers:
                    raise ValueError('Tracer {} is not supported for model {}.'.format(tracer, self.model_options.model_name))
        else:
            tracers = self.model_options.tracers
        return tracers
        

    ## access to model values (auxiliary)

    def _interpolate(self, data, interpolation_points, use_cache=False):
        from .constants import MODEL_INTERPOLATOR_FILE, MODEL_INTERPOLATOR_AMOUNT_OF_WRAP_AROUND, MODEL_INTERPOLATOR_NUMBER_OF_LINEAR_INTERPOLATOR, MODEL_INTERPOLATOR_SINGLE_OVERLAPPING_AMOUNT_OF_LINEAR_INTERPOLATOR, METOS_DIM

        data_points = data[:,:-1]
        data_values = data[:,-1]
        interpolator_file = MODEL_INTERPOLATOR_FILE

        ## try to get cached interpolator
        interpolator = self._cached_interpolator
        if interpolator is not None:
            interpolator.data_values = data_values
            logger.debug('Returning cached interpolator.')
        else:
            ## otherwise try to get saved interpolator
            if use_cache and os.path.exists(interpolator_file):
                interpolator = util.math.interpolate.Interpolator_Base.load(interpolator_file)
                interpolator.data_values = data_values
                logger.debug('Returning interpolator loaded from {}.'.format(interpolator_file))
            ## if no interpolator exists, create new interpolator
            else:
                interpolator = util.math.interpolate.Periodic_Interpolator(data_points=data_points, data_values=data_values, point_range_size=METOS_DIM, scaling_values=(METOS_DIM[1]/METOS_DIM[0], None, None, None), wrap_around_amount=MODEL_INTERPOLATOR_AMOUNT_OF_WRAP_AROUND, number_of_linear_interpolators=MODEL_INTERPOLATOR_NUMBER_OF_LINEAR_INTERPOLATOR, single_overlapping_amount_linear_interpolators=MODEL_INTERPOLATOR_SINGLE_OVERLAPPING_AMOUNT_OF_LINEAR_INTERPOLATOR)
                logger.debug('Returning new created interpolator.')

            self._cached_interpolator = interpolator

        ## interpolate
        interpolated_values = interpolator.interpolate(interpolation_points)

        ## save interpolate if cache used
        if use_cache and not os.path.exists(interpolator_file):
            interpolator.save(interpolator_file)

        ## return interpolated values
        assert not np.any(np.isnan(interpolated_values))
        return interpolated_values
    
    
    def _trajectory_with_load_function(self, trajectory_load_function, run_dir, model_parameters, tracers=None):
        TMP_DIR = simulation.model.constants.DATABASE_TMP_DIR

        assert callable(trajectory_load_function)
        tracers = self.check_tracers(tracers)

        trajectory_values = {}
        
        ## create and read trajectory
        if len(tracers) > 0:
            
            ## create trajectory
            if TMP_DIR is not None:
                tmp_dir = TMP_DIR
                os.makedirs(tmp_dir, exist_ok=True)
            else:
                tmp_dir = run_dir
    
            ## write trajectory
            trajectory_dir = tempfile.mkdtemp(dir=tmp_dir, prefix='trajectory_tmp_')
    
            tracer_input_filenames = ['{}_output.petsc'.format(tracer) for tracer in self.model_options.tracers]
            tracer_input_files = [os.path.join(run_dir, tracer_input_file) for tracer_input_file in tracer_input_filenames]
            self.start_run(model_parameters, trajectory_dir, years=1, tolerance=0, job_options=self.job_options_for_kind('trajectory'), tracer_input_files=tracer_input_files, write_trajectory=True, make_read_only=False)
    
            ## read trajectory        
            trajectory_output_dir = os.path.join(trajectory_dir, 'trajectory')
            for tracer in tracers:
                trajectory_values_tracer = trajectory_load_function(trajectory_output_dir, tracer=tracer)
                trajectory_values[tracer] = trajectory_values_tracer
    
            ## remove trajectory
            util.io.fs.remove_recursively(trajectory_dir, not_exist_okay=True, exclude_dir=False)

        ## return
        assert len(trajectory_values) == len(tracers)
        return trajectory_values


    def _trajectory_load_function_for_all(self, time_dim):
        trajectory_load_function = lambda trajectory_path, tracer: simulation.model.data.load_trajectories_to_map(trajectory_path, tracer, time_dim_desired=time_dim)
        return trajectory_load_function
    

    def _trajectory_load_function_for_points(self, points):
        from .constants import MODEL_INTERPOLATOR_NUMBER_OF_LINEAR_INTERPOLATOR

        ## convert points to map indices
        interpolation_points_dict = {}
            
        ## preprare interpolation points for each tracer
        for tracer, points_for_tracer in points.items():
            logger.debug('Calculating model output for tracer {} at {} points.'.format(tracer, len(points_for_tracer)))
            
            ## check tracer and points
            if tracer not in self.model_options.tracers:
                raise ValueError('Tracer {} is not supported for model {}.'.format(tracer, self.model_options.model_name))
            points_for_tracer = np.asanyarray(points_for_tracer)

            ## convert interpolation points to map indices
            if len(points_for_tracer) > 0:            
                interpolation_points_for_tracer = self.model_lsm.coordinates_to_map_indices(points_for_tracer, discard_year=True, int_indices=False)
                assert interpolation_points_for_tracer.ndim == 2 and interpolation_points_for_tracer.shape[1] == 4
                
                if MODEL_INTERPOLATOR_NUMBER_OF_LINEAR_INTERPOLATOR > 0:
                    for value_min, index in ([np.where(self.model_lsm.lsm > 0)[1].min(), 2], [0, 3]):
                        for k in range(len(interpolation_points_for_tracer)):
                            if interpolation_points_for_tracer[k, index] < value_min:
                                interpolation_points_for_tracer[k, index] = value_min
                    for value_max, index in ([np.where(self.model_lsm.lsm > 0)[1].max(), 2], [self.model_lsm.z_dim - 1, 3]):
                        for k in range(len(interpolation_points_for_tracer)):
                            if interpolation_points_for_tracer[k, index] > value_max:
                                interpolation_points_for_tracer[k, index] = value_max
                
                interpolation_points_dict[tracer] = interpolation_points_for_tracer
                

        ## interpolate trajectory function
        def interpolate_trajectory(trajectory_path, tracer):
            
            ## check if points for tracer are available
            try:
                interpolation_points_for_tracer = interpolation_points_dict[tracer]
            except KeyError:
                return np.empty([0,1])
            
            ## interpolate if points for tracer are available
            else:
                tracer_trajectory = simulation.model.data.load_trajectories_to_map_index_array(trajectory_path, tracers=tracer)
                interpolated_values_for_tracer = self._interpolate(tracer_trajectory, interpolation_points_for_tracer)
                return interpolated_values_for_tracer
        
        return interpolate_trajectory
    

    def _merge_data_sets(self, tracer_dict, concatenate_axis=0):
        tracer_merged_dict = {}
        tracer_split_dict = {}
        
        for tracer, tracer_value in tracer_dict.items():
            ## check if contains data set dict
            try:
                tracer_value.items
            ## use value, if no data set dict
            except AttributeError:
                tracer_value = np.asanyarray(tracer_value)
            ## merge all data set values to one array, else
            else:
                start_index = 0
                data_set_split_dict = {}
                data_set_values_list = []
                
                for data_set_name, data_set_value in tracer_value.items():
                    data_set_value = np.asanyarray(data_set_value)
                    data_set_values_list.append(data_set_value)
                    end_index = start_index + len(data_set_value)
                    data_set_split_slice = (slice(None),)*concatenate_axis + (slice(start_index, end_index),)
                    data_set_split_dict[data_set_name] = data_set_split_slice
                    start_index = end_index
                
                tracer_split_dict[tracer] = data_set_split_dict
                tracer_value = np.concatenate(data_set_values_list, axis=concatenate_axis)
                assert len(tracer_value) == end_index
            
            tracer_merged_dict[tracer] = tracer_value
        
        logger.debug('Merged data sets with tracer_split_dict {}.'.format(tracer_split_dict))
        return tracer_merged_dict, tracer_split_dict


    def _split_data_sets(self, tracer_dict, tracer_split_dict):
        tracer_splitted_dict = {}
        
        for tracer, tracer_value in tracer_dict.items():
            ## check if value was splitted
            try:
                data_set_split_dict = tracer_split_dict[tracer]
            ## use value, if not
            except KeyError:
                tracer_splitted_dict[tracer] = tracer_value
            ## split in data set values else
            else:
                data_set_dict = {}
                for data_set_name, data_set_split_slice in data_set_split_dict.items():
                    data_set_dict[data_set_name] = tracer_value[data_set_split_slice]
                assert sum(map(len, data_set_dict.values())) == len(tracer_value)
                tracer_splitted_dict[tracer] = data_set_dict
        
        logger.debug('Splitted data sets with tracer_split_dict {}.'.format(tracer_split_dict))
        return tracer_splitted_dict


    def _f(self, trajectory_load_function, tracers=None):
        tracers = self.check_tracers(tracers)
        matching_run_dir = self.run_dir
        model_parameters = self.model_options.parameters
        f = self._trajectory_with_load_function(trajectory_load_function, matching_run_dir, model_parameters, tracers=tracers)

        assert f is not None        
        assert len(f) == len(tracers)
        return f
    
    
    ## access to model values

    def f_all(self, time_dim, tracers=None):
        
        logger.debug('Calculating all f values for tracers {} with time dimension {}.'.format(tracers, time_dim))
        f = self._f(self._trajectory_load_function_for_all(time_dim), tracers=tracers)
        
        return f


    def f_points(self, points):
        logger.debug('Calculating f values at points for tracers {}.'.format(tuple(points.keys())))
        
        tracers = points.keys()
        points, split_dict = self._merge_data_sets(points)
        f = self._f(self._trajectory_load_function_for_points(points), tracers=tracers)
        f = self._split_data_sets(f, split_dict)
        
        return f
    

    def f_measurements(self, *measurements_list):
        logger.debug('Calculating f values for measurements {}.'.format(tuple(map(str, measurements_list))))
        measurements_collection = measurements.universal.data.MeasurementsCollection(*measurements_list)
        points_dict = measurements_collection.points_dict
        return self.f_points(points_dict)
        
        



class Model_With_F_And_DF(Model_With_F):

    
    @property
    def derivative_dir(self):
        step_size = self.model_options.derivative_options.step_size
        derivative_dir = os.path.join(self.parameter_set_dir, simulation.model.constants.DATABASE_DERIVATIVE_DIRNAME.format(step_size=step_size))
        logger.debug('Returning derivative directory {}.'.format(derivative_dir))
        return spinup_dir
    

    def _df(self, trajectory_load_function, partial_derivative_kind, tracers=None):
        ## check tracers
        tracers = self.check_tracers(tracers)
        
        ## return empty array if no tracer wanted
        if len(tracers) == 0:
            return {}

        ## apply partial_derivative_kind
        if partial_derivative_kind == 'model_parameters':
            def convert_partial_derivative_parameters_to_start_run_parameters(partial_derivative_parameters):
                assert len(partial_derivative_parameters) == self.model_options.parameters_len
                return {'model_parameters': partial_derivative_parameters, 'total_concentration_factor': 1}
        
            partial_derivative_parameters_bounds = self.model_options.parameters_bounds
            partial_derivative_parameters_typical_values =  self.model_options.derivative_options.parameters_typical_values
            partial_derivative_parameters_undisturbed = self.model_options.parameters
                
        elif partial_derivative_kind == 'total_concentration_factor':
            def convert_partial_derivative_parameters_to_start_run_parameters(partial_derivative_parameters):
                assert len(partial_derivative_parameters) == 1
                return {'model_parameters': self.model_options.parameters, 'total_concentration_factor': partial_derivative_parameters[0]}
        
            partial_derivative_parameters_bounds = np.array([[0,np.inf]])
            partial_derivative_parameters_typical_values =  np.array([1])
            partial_derivative_parameters_undisturbed = np.array([1])
        
        else:
            raise ValueError('Partial derivative kind {} is not supported.'.format(partial_derivative_kind))
        
        ## get needed model options
        MODEL_DERIVATIVE_SPINUP_YEARS = self.model_options.derivative_options.years
        MODEL_DERIVATIVE_STEP_SIZE = self.model_options.derivative_options.step_size
        MODEL_DERIVATIVE_ACCURACY_ORDER = self.model_options.derivative_options.accuracy_order
        spinup_options = self.model_options.spinup_options

        
        ## get direavtive dir and spinup run dir
        derivative_dir = self.derivative_dir
        spinup_matching_run_dir = self.matching_run_dir(spinup_options)
        spinup_matching_run_years = self.get_total_years(spinup_matching_run_dir)


        ## get f if accuracy_order is 1
        if MODEL_DERIVATIVE_ACCURACY_ORDER == 1:
            spinup_options_f = {'years':spinup_matching_run_years + MODEL_DERIVATIVE_SPINUP_YEARS, 'tolerance':0, 'combination':'or'}
            spinup_options_f = simulation.model.options.SpinupOptions(spinup_options_f)
            self.model_options.spinup_options = spinup_options_f
            f_parameters = self._f(trajectory_load_function)
            self.model_options.spinup_options = spinup_options
        else:
            f_parameters = None
        
        
        ## define evaluation functions for finite differences

        job_options = self.job_options_for_kind('derivative')
        partial_derivative_run_dirs = {}
        
        def start_partial_derivative_run(partial_derivative_parameters):
            parameter_index = np.where(partial_derivative_parameters != partial_derivative_parameters_undisturbed)[0]
            assert len(parameter_index) == 1
            parameter_index = parameter_index[0]
            h = partial_derivative_parameters[parameter_index] - partial_derivative_parameters_undisturbed[parameter_index]
            h_factor = int(np.sign(h))
    
            ## get run dir
            partial_derivative_dirname = simulation.model.constants.DATABASE_PARTIAL_DERIVATIVE_DIRNAME.format(kind=partial_derivative_kind, index=parameter_index, h_factor=h_factor)
            partial_derivative_dir = os.path.join(derivative_dir, partial_derivative_dirname)
            partial_derivative_run_dir = self.last_run_dir(partial_derivative_dir)
            logger.debug('Checking partial derivative runs in {}.'.format(partial_derivative_dir))
            
            ## get corresponding spinup run dir
            if partial_derivative_run_dir is not None:
                try:
                    with simulation.model.job.Metos3D_Job(partial_derivative_run_dir, force_load=True) as job:
                        partial_derivative_spinup_run_tracer_input_files = job.model_tracer_input_files
                    partial_derivative_spinup_run_dir = [os.path.dirname(partial_derivative_spinup_run_tracer_input_file) for partial_derivative_spinup_run_tracer_input_file in partial_derivative_spinup_run_tracer_input_files]
                    assert all([partial_derivative_spinup_run_dir[0] == a for a in partial_derivative_spinup_run_dir[1:]])
                    partial_derivative_spinup_run_dir = partial_derivative_spinup_run_dir[0]

                except OSError:
                    partial_derivative_spinup_run_dir = None

            ## make new run if run not matching
            if not self.is_run_matching_options(partial_derivative_run_dir, {'years':MODEL_DERIVATIVE_SPINUP_YEARS, 'tolerance':0, 'combination':'or'}) or not self.is_run_matching_options(partial_derivative_spinup_run_dir, spinup_options):   
                
                ## remove old run
                if partial_derivative_run_dir is not None:
                    logger.debug('Old partial derivative run {} is not matching desired option. It is removed.'.format(partial_derivative_run_dir))
                    util.io.fs.remove_recursively(partial_derivative_run_dir, not_exist_okay=True, exclude_dir=False)
                
                ## create new run dir
                partial_derivative_run_dir = self.make_new_run_dir(partial_derivative_dir)   
                
                ## if no job setup available, get best job setup
                if job_options['nodes_setup'] is None:
                    job_options['nodes_setup'] = util.batch.universal.system.NodeSetup(memory=simulation.model.constants.JOB_MEMORY_GB)
                    
                ## get tracer input files
                spinup_matching_run_dir_with_env = spinup_matching_run_dir.replace(simulation.constants.SIMULATION_OUTPUT_DIR, '${{{}}}'.format(simulation.constants.SIMULATION_OUTPUT_DIR_ENV_NAME))
                tracer_input_filenames = ['{}_output.petsc'.format(tracer) for tracer in self.model_options.tracers]
                tracer_input_files = [os.path.join(spinup_matching_run_dir_with_env, tracer_input_filename) for tracer_input_filename in tracer_input_filenames]

                ## start job
                start_run_parameters_dict = convert_partial_derivative_parameters_to_start_run_parameters(partial_derivative_parameters)
                partial_derivative_model_parameters = start_run_parameters_dict['model_parameters']
                total_concentration_factor = start_run_parameters_dict['total_concentration_factor']
                self.start_run(partial_derivative_model_parameters, partial_derivative_run_dir, MODEL_DERIVATIVE_SPINUP_YEARS, tolerance=0, job_options=job_options, tracer_input_files=tracer_input_files, wait_until_finished=False, total_concentration_factor=total_concentration_factor)
                
            partial_derivative_run_dirs[tuple(partial_derivative_parameters)] = partial_derivative_run_dir
            
            return 0
            

        tracer_start_stop_indices = [0]
        
        def get_partial_derivative_run_value(partial_derivative_parameters):
            ## wait partial derivative run to finish
            partial_derivative_run_dir = partial_derivative_run_dirs[tuple(partial_derivative_parameters)]
            self.wait_until_run_finished(partial_derivative_run_dir)

            ## get trajectory
            partial_derivative_model_parameters = convert_partial_derivative_parameters_to_start_run_parameters(partial_derivative_parameters)['model_parameters']
            trajectory_dict = self._trajectory_with_load_function(trajectory_load_function, partial_derivative_run_dir, partial_derivative_model_parameters)
            trajectory_list = [trajectory_dict[tracer] for tracer in tracers]
            
            ## store length of each tracer
            if len(tracer_start_stop_indices) == 1:
                start_index = 0
                for trajectory in trajectory_list:
                    stop_index = start_index + len(trajectory)
                    tracer_start_stop_indices.append(stop_index)
                    start_index = stop_index
            
            ## concatenate and return
            trajectory = np.concatenate(trajectory_list)
            
            return trajectory

        
        ## calculate deviation
        for function in (start_partial_derivative_run, get_partial_derivative_run_value):
            df_concatenated = util.math.finite_differences.calculate(function, partial_derivative_parameters_undisturbed, f_x=f_parameters, typical_x=partial_derivative_parameters_typical_values, bounds=partial_derivative_parameters_bounds, accuracy_order=MODEL_DERIVATIVE_ACCURACY_ORDER, eps=MODEL_DERIVATIVE_STEP_SIZE, use_always_typical_x=True)
        df_concatenated = np.moveaxis(df_concatenated, 0, -1)
        
        ## unpack concatenation
        logger.debug('Unpacking derivative with shape {} for tracers with tracer_start_stop_indices {}.'.format(df_concatenated.shape, tracer_start_stop_indices))
        assert len(tracer_start_stop_indices) == len(tracers) + 1
        assert max(tracer_start_stop_indices) == len(df_concatenated)
        
        df = {}
        for tracer_index in range(len(tracers)):
            df_tracer = df_concatenated[tracer_start_stop_indices[tracer_index] : tracer_start_stop_indices[tracer_index+1]]
            tracer = tracers[tracer_index]
            df[tracer] = df_tracer
        
        ## return
        assert len(df) == len(tracers)
        return df


    ## access to model values

    def df_all(self, time_dim, tracers=None, partial_derivative_kind='model_parameters'):
        tracers = self.check_tracers(tracers)
        
        logger.debug('Calculating all df values for tracers {} with time dimension {} and partial_derivative_kind {}.'.format(tracers, time_dim, partial_derivative_kind))
        
        df = self._df(self._trajectory_load_function_for_all(time_dim=time_dim), partial_derivative_kind=partial_derivative_kind, tracers=tracers)
        return df


    def df_points(self, points, partial_derivative_kind='model_parameters'):
        logger.debug('Calculating df values at points {} and partial_derivative_kind {}.'.format(tuple(map(len, points)), partial_derivative_kind))
        
        tracers = points.keys()
        points, split_dict = self._merge_data_sets(points)
        df = self._df(self._trajectory_load_function_for_points(points), partial_derivative_kind=partial_derivative_kind, tracers=tracers)
        df = self._split_data_sets(df, split_dict)
        
        return df
    

    def df_measurements(self, *measurements_list, partial_derivative_kind='model_parameters'):
        logger.debug('Calculating df values for measurements {} and partial_derivative_kind {}.'.format(tuple(map(str, measurements_list)), partial_derivative_kind))
        
        measurements_collection = measurements.universal.data.MeasurementsCollection(*measurements_list)
        points_dict = measurements_collection.points_dict
        
        return self.df_points(points_dict, partial_derivative_kind=partial_derivative_kind)




## Cached versions


class Model_Database_MemoryCached(Model_Database):
    
    @property
    @util.cache.memory_based.decorator(dependency=('self.database_output_dir', 'self.model_options.model_name'))
    def model_dir(self):
        return super().model_dir
    
    @property
    @util.cache.memory_based.decorator(dependency=('self.model_dir', 'self.model_options.initial_concentration_options.use_constant_concentrations'))
    def initial_concentration_base_dir(self):
        return super().initial_concentration_base_dir
    
    @property
    @util.cache.memory_based.decorator(dependency=('self.model_dir', 'self.model_options.initial_concentration_options.tolerance_options.relative',  'self.model_options.initial_concentration_options.tolerance_options.absolute'))
    def _constant_concentrations_db(self):
        return super()._constant_concentrations_db
    
    @property
    @util.cache.memory_based.decorator(dependency=('self.model_dir', 'self.model_options.tracers', 'self.model_options.initial_concentration_options.tolerance_options.relative',  'self.model_options.initial_concentration_options.tolerance_options.absolute'))
    def _vector_concentrations_db(self):
        return super()._vector_concentrations_db

    @property
    @util.cache.memory_based.decorator(dependency=('self.model_dir', 'self.model_options.tracers', 'self.model_options.initial_concentration_options.concentrations', 'self.model_options.initial_concentration_options.tolerance_options.relative',  'self.model_options.initial_concentration_options.tolerance_options.absolute'))
    def initial_concentration_dir_index(self):
        return super().initial_concentration_dir_index
    
    @property
    @util.cache.memory_based.decorator(dependency=('self.initial_concentration_base_dir', 'self.initial_concentration_dir_index'))
    def initial_concentration_dir(self):
        return super().initial_concentration_dir
    
    @property
    @util.cache.memory_based.decorator(dependency=('self.initial_concentration_dir_index', 'self.model_dir', 'self.model_options.tracers'))
    def initial_concentration_files(self):
        return super().initial_concentration_files
    
    @property
    @util.cache.memory_based.decorator(dependency=('self.initial_concentration_dir', 'self.model_options.time_step'))
    def time_step_dir(self):
        return super().time_step_dir
    
    @property
    @util.cache.memory_based.decorator(dependency=('self.time_step_dir', 'self.model_options.parameter_tolerance_options.relative', 'self.model_options.parameter_tolerance_options.absolute'))
    def _parameter_db(self):
        return super()._parameter_db
    
    @property
    @util.cache.memory_based.decorator(dependency=('self.time_step_dir', 'self.model_options.parameters', 'self.model_options.parameter_tolerance_options.relative', 'self.model_options.parameter_tolerance_options.absolute'))
    def parameter_set_dir(self):
        return super().parameter_set_dir
    
    @property
    @util.cache.memory_based.decorator(dependency=('self.time_step_dir', 'self.model_options.parameters', 'self.model_options.parameter_tolerance_options.relative', 'self.model_options.parameter_tolerance_options.absolute'))
    def closest_parameter_set_dir(self):
        return super().closest_parameter_set_dir
    
    @property
    @util.cache.memory_based.decorator(dependency='self.parameter_set_dir')
    def spinup_dir(self):
        return super().spinup_dir
    
    @property
    @util.cache.memory_based.decorator(dependency='self.closest_parameter_set_dir')
    def closest_spinup_dir(self):
        return super().closest_spinup_dir
    
    @property
    @util.cache.memory_based.decorator(dependency=('self.spinup_dir', 'self.model_options.spinup_options.years', 'self.model_options.spinup_options.tolerance', 'self.model_options.spinup_options.combination'))
    def run_dir(self):
        return super().run_dir




class Model_With_F_MemoryCached(Model_Database_MemoryCached, Model_With_F):
    pass




class Model_With_F_And_DF_MemoryCached(Model_With_F_MemoryCached, Model_With_F_And_DF):

    @property
    @util.cache.memory_based.decorator(dependency=('self.parameter_set_dir', 'self.model_options.derivative_options.step_size'))
    def derivative_dir(self):
        step_size = self.model_options.derivative_options.step_size
        derivative_dir = os.path.join(self.parameter_set_dir, simulation.model.constants.DATABASE_DERIVATIVE_DIRNAME.format(step_size=step_size))
        logger.debug('Returning derivative directory {}.'.format(derivative_dir))
        return derivative_dir




Model = Model_With_F_And_DF_MemoryCached




class DatabaseError(Exception):
    def __init__(self, database, message):
        self.database = database
        message = 'Error at database {}: {}'.format(self.database_output_dir, message)
        super().__init__(message)

