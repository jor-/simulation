import os
import stat
import tempfile
import warnings

import numpy as np

import ndop.model.data
import ndop.model.job

import measurements.util.interpolate

import util.io.fs
import util.pattern
import util.math.interpolate

import util.logging
logger = util.logging.logger


# spinup_options: years, tolerance=0, combination='or'

class Model():

    def __init__(self, job_setup=None, default_spinup_options={'years':10000, 'tolerance':0.0, 'combination':'or'}, df_options= {'years':100, 'step_size': 10**(-7)}):
        from .constants import MODEL_PARAMETER_LOWER_BOUND, MODEL_PARAMETER_UPPER_BOUND, MODEL_OUTPUT_DIR, METOS_TRACER_DIM

        logger.debug('Model initiated with job setup {}, default spinup options {} and df options.'.format(job_setup, default_spinup_options, df_options))

        ## job setup collection
        # convert job setup to job setup collection
        if job_setup is None:
            job_setup = {}

        job_setup_collection = {}
        keys = list(job_setup.keys())
        kinds = ['spinup', 'derivative', 'trajectory']
        if any(kind in keys for kind in kinds):
            job_setup_collection = job_setup
        else:
            job_setup_collection['spinup'] = job_setup

        # if not passed, use default job setups
        try:
            job_setup_collection['spinup']
        except KeyError:
            job_setup_collection['spinup'] = {}
        try:
            job_setup_collection['derivative']
        except KeyError:
            job_setup_collection['derivative'] = job_setup_collection['spinup'].copy()
            job_setup_collection['derivative']['nodes_setup'] = None
        try:
            job_setup_collection['trajectory']
        except KeyError:
            job_setup_collection['trajectory'] = job_setup_collection['derivative'].copy()
            job_setup_collection['trajectory']['nodes_setup'] = None

        # if no name passed, use default name
        try:
            default_name = job_setup['name']
        except KeyError:
            default_name = ''
        for kind in kinds:
            try:
                job_setup_collection[kind]['name']
            except KeyError:
                job_setup_collection[kind]['name'] = default_name

        self.job_setup_collection = job_setup_collection



#         ## job options
#         if job_setup is not None:
#             self.job_setup_spinup = job_setup_spinup.copy()
#         else:
#             self.job_setup_spinup = {}
#
#         self.job_setup_best = self.job_setup_spinup.copy()
#         self.job_setup_best['nodes_setup'] = None
#
#         if jop_setup_derivative is not None:
#             self.jop_setup_derivative = jop_setup_derivative.copy()
#         else:
#             self.jop_setup_derivative = self.job_setup_best.copy()

        ## default spinup options
        self.default_spinup_options = default_spinup_options

        ## df options
        self.df_options = df_options

        ## parameter bounds
        self.parameters_lower_bound = MODEL_PARAMETER_LOWER_BOUND
        self.parameters_upper_bound = MODEL_PARAMETER_UPPER_BOUND

        ## model output dir
        self.model_output_dir = MODEL_OUTPUT_DIR

        ## empty interpolator cache
        self._interpolator_cached = None



    def check_if_parameters_in_bounds(self, parameters):
        if any(parameters < self.parameters_lower_bound):
            indices = np.where(parameters < self.parameters_lower_bound)
#             string = ''
#             for i in indices:
#                 string += str(parameters[i]) + '<' + str(self.parameters_lower_bound[i])
#                 if i < len(indices) - 1:
#                     string += ', '
#             raise ValueError('Some parameters are lower than the lower bound: ' + string)
            raise ValueError('The parameters {} are not allowed. The parameters with the indices {} are below their lower bound {}.'.format(parameters, indices, self.parameters_lower_bound[indices]))

        if any(parameters > self.parameters_upper_bound):
            indices = np.where(parameters > self.parameters_upper_bound)
#             string = ''
#             for i in indices:
#                 string += str(parameters[i]) + '>' + str(self.parameters_upper_bound[i])
#                 if i < len(indices) - 1:
#                     string += ', '
#             raise ValueError('Some parameters are upper than the upper bound: ' + string)
            raise ValueError('The parameters {} are not allowed. The parameters with the indices {} are above their upper bound {}.'.format(parameters, indices, self.parameters_upper_bound[indices]))



    def get_job_setup(self, kind):
        job_setup = self.job_setup_collection[kind]
        job_setup = job_setup.copy()
        job_setup['nodes_setup'] = job_setup['nodes_setup'].copy()
        return job_setup
        


    def get_parameters_diff(self, parameters, parameter_set_dir):
        from .constants import MODEL_PARAMETERS_FILENAME, MODEL_PARAMETERS_MAX_REL_DIFF, MODEL_PARAMETERS_MAX_ABS_DIFF

        parameters_file = os.path.join(parameter_set_dir, MODEL_PARAMETERS_FILENAME)
        current_parameters = np.loadtxt(parameters_file)

        parameter_diffs = np.abs(parameters - current_parameters) / np.maximum(np.abs(parameters), MODEL_PARAMETERS_MAX_ABS_DIFF/MODEL_PARAMETERS_MAX_REL_DIFF)
        parameter_diff = parameter_diffs.max()

        return parameter_diff





    ## access to dirs

    def get_time_step_dir(self, time_step, create=True):
        from .constants import MODEL_TIME_STEP_DIRNAME

        ## get directory for time step
        model_output_dir = self.model_output_dir
        logger.debug('Searching for directory for time step size {} in {}.'.format(time_step, model_output_dir))

        time_step_dirname = MODEL_TIME_STEP_DIRNAME.format(time_step)
        time_step_dir = os.path.join(model_output_dir, time_step_dirname)

        if os.path.exists(time_step_dir):
            logger.debug('Directory {} for time step found.'.format(time_step_dir))
        else:
            if create:
                os.makedirs(time_step_dir)
                logger.debug('Directory {} for time step created.'.format(time_step_dir))
            else:
                time_step_dir = None
                logger.debug('No directory for time step found. Non created.')

        return time_step_dir



    def get_closest_parameter_set_dir(self, time_step, parameters, no_spinup_okay=True):
        ## check input
        self.check_if_parameters_in_bounds(parameters)

        ## get directory for time step
        time_step_dir = self.get_time_step_dir(time_step, create=False)

        closest_parameter_set_dir = None
        closest_parameters_diff = float('inf')

        ## if time step dir exists, search for matching parameter set
        if time_step_dir is not None:
            from .constants import MODEL_SPINUP_DIRNAME

            logger.debug('Searching for directory for parameters as close as possible to {} in {}.'.format(parameters, time_step_dir))

            ## search for parameter set directories with nearby parameters
            parameter_set_number = 0

            parameter_set_dirs = util.io.fs.get_dirs(time_step_dir)
            number_of_parameter_set_dirs = len(parameter_set_dirs)
            while closest_parameters_diff > 0 and parameter_set_number < number_of_parameter_set_dirs:
                current_parameter_set_dir = parameter_set_dirs[parameter_set_number]
                try:
                    current_parameters_diff = self.get_parameters_diff(parameters, current_parameter_set_dir)
                except (OSError, IOError):
                    warnings.warn('Could not read the parameters file in {}!'.format(current_parameter_set_dir))
                    current_parameters_diff = float('inf')

                if current_parameters_diff < closest_parameters_diff:
                    better = False
                    if not no_spinup_okay:
                        current_spinup_dir = os.path.join(current_parameter_set_dir, MODEL_SPINUP_DIRNAME)
                        last_run_dir = self.get_last_run_dir(current_spinup_dir)
                        if last_run_dir is not None:
                            better = True
                    else:
                        better = True

                    if better:
                        closest_parameter_set_dir = current_parameter_set_dir
                        closest_parameters_diff = current_parameters_diff

                parameter_set_number += 1

        if closest_parameter_set_dir is not None:
            logger.debug('Closest parameter set dir is {}.'.format(closest_parameter_set_dir))
        else:
            logger.debug('No closest parameter set dir found.')

        return closest_parameter_set_dir




    def get_parameter_set_dir(self, time_step, parameters, create=True):
        from .constants import MODEL_PARAMETERS_MAX_REL_DIFF

        ## search for directories with matching parameters
        logger.debug('Searching parameter directory for time_step {} and parameters {} with create {}.'.format(time_step, parameters, create))

        parameter_set_dir = self.get_closest_parameter_set_dir(time_step, parameters, no_spinup_okay=True)
        if parameter_set_dir is not None:
            parameters_diff = self.get_parameters_diff(parameters, parameter_set_dir)
            if parameters_diff > MODEL_PARAMETERS_MAX_REL_DIFF:
                parameter_set_dir = None


        ## make new model_output if the parameters are not matching
        if parameter_set_dir is None:
            if create:
                from .constants import MODEL_PARAMETERS_SET_DIRNAME, MODEL_PARAMETERS_FILENAME, MODEL_PARAMETERS_FORMAT_STRING_OLD_STYLE

                ## create time_step_dir
                time_step_dir = self.get_time_step_dir(time_step, create=True)
                
                ## create parameter_set_dir
                free_parameter_set_dir_number = 0
                created = False
                found = False
                while not created:
                    ## search free index
                    while not found:
                        parameter_set_dirname = MODEL_PARAMETERS_SET_DIRNAME.format(free_parameter_set_dir_number)
                        parameter_set_dir = os.path.join(time_step_dir, parameter_set_dirname)
                        found = not os.path.exists(parameter_set_dir)
                        free_parameter_set_dir_number += 1
    
                    ## make dir
                    try:
                        os.makedirs(parameter_set_dir)
                        created = True
                    except FileExistsError:
                        found = False

                ## create parameters_file
                parameters_file = os.path.join(parameter_set_dir, MODEL_PARAMETERS_FILENAME)
                np.savetxt(parameters_file, parameters, fmt=MODEL_PARAMETERS_FORMAT_STRING_OLD_STYLE)
                os.chmod(parameters_file, stat.S_IRUSR)

                logger.debug('Directory for parameters created in {}.'.format(parameter_set_dir))
            else:
                logger.debug('No matching directory for parameters found. Non created.')
        else:
            logger.debug('Matching directory for parameters found at {}.'.format(parameter_set_dir))

        assert parameter_set_dir is not None or not create
        return parameter_set_dir


    def get_spinup_run_dir(self, parameter_set_dir, spinup_options, start_from_closest_parameters=False):
        from .constants import MODEL_SPINUP_DIRNAME, MODEL_PARAMETERS_FILENAME, MODEL_SPINUP_MAX_YEARS

        spinup_dir = os.path.join(parameter_set_dir, MODEL_SPINUP_DIRNAME)

        logger.debug('Searching for spinup with options {} in {}.'.format(spinup_options, spinup_dir))

        last_run_dir = self.get_last_run_dir(spinup_dir)

        ## matching spinup found
        if self.is_run_matching_options(last_run_dir, spinup_options):
            run_dir = last_run_dir
            logger.debug('Matching spinup found at {}.'.format(last_run_dir))

        ## create new spinup
        else:
            logger.debug('No matching spinup found.')

            ## get parameters
            parameter_file = os.path.join(parameter_set_dir, MODEL_PARAMETERS_FILENAME)
            parameters = np.loadtxt(parameter_file)

            ## get time_step
            time_step_dirname = os.path.basename(os.path.dirname(parameter_set_dir))
            time_step = util.pattern.get_int_in_string(time_step_dirname)

            ## no previous run exists and starting from closest parameters get last run from closest parameters
            if last_run_dir is None and start_from_closest_parameters:
                closest_parameter_set_dir = self.get_closest_parameter_set_dir(time_step, parameters, no_spinup_okay=False)
                closest_spinup_dir = os.path.join(closest_parameter_set_dir, MODEL_SPINUP_DIRNAME)
                last_run_dir = self.get_last_run_dir(closest_spinup_dir)

            ## finish last run
            if last_run_dir is not None:
                self.wait_until_job_finished(last_run_dir)

            ## make new run
            years, tolerance, combination = self.all_spinup_options(spinup_options)

            if combination == 'or':
                run_dir = self.make_run(spinup_dir, parameters, years, tolerance, time_step, self.get_job_setup('spinup'), tracer_input_path=last_run_dir)
            elif combination == 'and':
                run_dir = self.get_spinup_run_dir(parameter_set_dir, {'years':years, 'tolerance':0, 'combination':'or'}, start_from_closest_parameters)
                run_dir = self.get_spinup_run_dir(parameter_set_dir, {'years':MODEL_SPINUP_MAX_YEARS, 'tolerance':tolerance, 'combination':'or'}, start_from_closest_parameters)

            logger.debug('Spinup directory created at {}.'.format(run_dir))

        return run_dir



    def make_run(self, output_path, parameters, years, tolerance, time_step, job_setup, tracer_input_path=None, wait_until_finished=True):
        from .constants import MODEL_RUN_DIRNAME, MODEL_RUN_OPTIONS_FILENAME

        ## check parameters
        self.check_if_parameters_in_bounds(parameters)

        ## get next run index
        util.io.fs.makedirs_if_not_exists(output_path)
        run_dirs = self.get_run_dirs(output_path)
        next_run_index = len(run_dirs)

        ## create run dir
        run_dirname = MODEL_RUN_DIRNAME.format(next_run_index)
        run_dir = os.path.join(output_path, run_dirname)

        logger.debug('Creating new run directory at {}.'.format(run_dir))

        os.makedirs(run_dir)

        ## create run options file
        run_options = np.array((years, tolerance, time_step))
        run_options_file = os.path.join(run_dir, MODEL_RUN_OPTIONS_FILENAME)
        np.savetxt(run_options_file, run_options)
        os.chmod(run_options_file, stat.S_IRUSR)

        ## create run
        if tracer_input_path is not None and output_path == os.path.dirname(tracer_input_path):
            last_years = self.get_total_years(tracer_input_path)
        else:
            last_years = 0

        self.run_job(parameters, run_dir, years-last_years, tolerance, time_step, job_setup, tracer_input_path=tracer_input_path, wait_until_finished=wait_until_finished)

        return run_dir




    ## run job

    def run_job(self, model_parameters, output_path, years, tolerance, time_step, job_setup, write_trajectory=False, tracer_input_path=None, make_read_only=True, wait_until_finished=True):
        from ndop.constants import BASE_DIR, BASE_DIR_ENV_NAME
        
        logger.debug('Running job with years {} tolerance {} time_step {} tracer_input_path {}.'.format(years, tolerance, time_step, tracer_input_path))
        assert years >= 0
        assert tolerance >= 0
        assert time_step >= 1

        ## check parameters
        self.check_if_parameters_in_bounds(model_parameters)

        ## execute job
        output_path_with_env = output_path.replace(BASE_DIR, '${{{}}}'.format(BASE_DIR_ENV_NAME))
        with ndop.model.job.Metos3D_Job(output_path_with_env) as job:
            job.write_job_file(model_parameters, years=years, tolerance=tolerance, time_step=time_step, write_trajectory=write_trajectory, tracer_input_path=tracer_input_path, job_setup=job_setup)
            job.start()
            job.make_read_only_input(make_read_only)

        ## wait to finish
        if wait_until_finished:
            self.wait_until_job_finished(output_path, make_read_only=make_read_only)
        else:
            logger.debug('Not waiting for job to finish.')


    def wait_until_job_finished(self, run_dir, make_read_only=True):
        with ndop.model.job.Metos3D_Job(run_dir, force_load=True) as job:
            job.make_read_only_input(make_read_only)
            job.wait_until_finished()
            job.make_read_only_output(make_read_only)



    ## access to spinup options

    def spinup_option(self, option, spinup_options=None):
        if spinup_options is not None:
            try:
                option_value = spinup_options[option]
            except KeyError:
                option_value = None

            if option is 'combination':
                valid_values = ('and', 'or')
                if option_value in valid_values:
                    return option_value
                else:
                    warnings.warn('The combination value {} is not valid. Only the values {} are supported.'.format(option_value, valid_values))
            else:
                return option_value

        return self.spinup_option(option, self.default_spinup_options)

    def all_spinup_options(self, spinup_options=None):
        all_spinup_options = []
        for option in ('years', 'tolerance', 'combination'):
            all_spinup_options.append(self.spinup_option(option, spinup_options))
        return all_spinup_options



    ## access to runs
    def is_run_matching_options(self, run_dir, spinup_options=None):
        from .constants import MODEL_SPINUP_MAX_YEARS

        years, tolerance, combination = self.all_spinup_options(spinup_options)

        if run_dir is not None:
            run_years = self.get_total_years(run_dir)
            run_tolerance = self.get_real_tolerance(run_dir)

            if combination == 'and':
                is_matching = (run_years >= years and run_tolerance <= tolerance) or run_years >= MODEL_SPINUP_MAX_YEARS
                if is_matching and run_tolerance > tolerance:
                    warnings.warn('The run {} does not match the desired tolerance {}, but the max spinup years {} are reached.'.format(run_dir, tolerance, MODEL_SPINUP_MAX_YEARS))
            elif combination == 'or':
                is_matching = (run_years >= years or run_tolerance <= tolerance)
            else:
                raise ValueError('Combination "{}" unknown.'.format(combination))

        else:
            is_matching = False

        if is_matching:
            logger.debug('Run in {} is matching spinup options {}.'.format(run_dir, spinup_options))
        else:
            logger.debug('Run in {} is not matching spinup options {}.'.format(run_dir, spinup_options))

        return is_matching



    def get_run_dirs(self, search_path):
        from .constants import MODEL_RUN_DIRNAME

        run_dir_condition = lambda file: os.path.isdir(file) and util.pattern.is_matching(os.path.basename(file), MODEL_RUN_DIRNAME)
        try:
            run_dirs = util.io.fs.filter_files(search_path, run_dir_condition)
        except (OSError, IOError) as exception:
            warnings.warn('It could not been searched in the search path "' + search_path + '": ' + str(exception))
            run_dirs = []

        return run_dirs



    def get_last_run_dir(self, search_path):
        logger.debug('Searching for last run in {}.'.format(search_path))

        run_dirs = self.get_run_dirs(search_path)

        last_run_dir = None
        last_run_index =  len(run_dirs) - 1

        while last_run_dir is None and last_run_index >= 0:
            last_run_dir = run_dirs[last_run_index]
            logger.debug('Searching in {}.'.format(last_run_dir))

            # check job options file
            if last_run_dir is not None:
                try:
                    with ndop.model.job.Metos3D_Job(last_run_dir, force_load=True) as job:
                        pass
                except (OSError, IOError) as exception:
                    warnings.warn('Could not read the job options file from "' + last_run_dir + '": ' + str(exception))
                    last_run_dir = None

            last_run_index -= 1

        logger.debug('Run {} found.'.format(last_run_dir))

        return last_run_dir



    def get_previous_run_dir(self, run_dir):
        from .constants import MODEL_RUN_DIRNAME

        (spinup_dir, run_dirname) = os.path.split(run_dir)
        run_index = util.pattern.get_int_in_string(run_dirname)
        if run_index > 0:
            previous_run_dirname = MODEL_RUN_DIRNAME.format(run_index - 1)
            previous_run_dir = os.path.join(spinup_dir, previous_run_dirname)
        else:
            previous_run_dir = None

        return previous_run_dir



    ##  access run properties
    def get_total_years(self, run_dir):
        total_years = 0

        while run_dir is not None:
            with ndop.model.job.Metos3D_Job(run_dir, force_load=True) as job:
                years = job.last_year
            total_years += years
            run_dir = self.get_previous_run_dir(run_dir)

        return total_years



    def get_real_tolerance(self, run_dir):
        with ndop.model.job.Metos3D_Job(run_dir, force_load=True) as job:
            tolerance = job.last_tolerance

        return tolerance



    def get_time_step(self, run_dir):
        with ndop.model.job.Metos3D_Job(run_dir, force_load=True) as job:
            time_step = job.time_step

        return time_step



    def get_tracer_input_dir(self, run_dir):
        with ndop.model.job.Metos3D_Job(run_dir, force_load=True) as job:
            tracer_input_dir = job.tracer_input_path

        return tracer_input_dir





    ## access to model values

    def _get_trajectory(self, load_trajectory_function, run_dir, parameters):
        from .constants import METOS_TRACER_DIM
        from util.constants import TMP_DIR

        assert callable(load_trajectory_function)

        run_time_step = self.get_time_step(run_dir)
        trajectory_values = ()

        ## create trajectory
        if TMP_DIR is not None:
            tmp_dir = TMP_DIR
        else:
            tmp_dir = run_dir

        ## write trajectory
        trajectory_dir = tempfile.mkdtemp(dir=tmp_dir, prefix='trajectory_tmp_')
        self.run_job(parameters, trajectory_dir, years=1, tolerance=0, time_step=run_time_step, job_setup=self.get_job_setup('trajectory'), tracer_input_path=run_dir, write_trajectory=True, make_read_only=False)

        ## read trajectory
        trajectory_output_dir = os.path.join(trajectory_dir, 'trajectory')
        for tracer_index in range(METOS_TRACER_DIM):
            tracer_trajectory_values = load_trajectory_function(trajectory_output_dir, tracer_index)
            trajectory_values += (tracer_trajectory_values,)

        ## remove trajectory
        util.io.fs.remove_recursively(trajectory_dir)

        ## return
        assert len(trajectory_values) == METOS_TRACER_DIM
        return trajectory_values


    def _get_load_trajectory_function_for_all(self, time_dim_desired):
        load_trajectory_function = lambda trajectory_path, tracer_index : ndop.model.data.load_trajectories_to_map(trajectory_path, tracer_index, time_dim_desired=time_dim_desired)
        return load_trajectory_function


        

    def _get_load_trajectory_function_for_points(self, points):
        from .constants import LSM, MODEL_INTERPOLATOR_NUMBER_OF_LINEAR_INTERPOLATOR

        ## convert to map indices
        interpolation_points = []
        for tracer_points in points:
            tracer_interpolation_points = np.array(tracer_points, copy=True)
            tracer_interpolation_points = LSM.coordinates_to_map_indices(tracer_interpolation_points)
            assert tracer_interpolation_points.ndim == 2 and tracer_interpolation_points.shape[1] == 4
            
            if MODEL_INTERPOLATOR_NUMBER_OF_LINEAR_INTERPOLATOR > 0:
                for value_min, index in ([np.where(LSM.lsm > 0)[1].min(), 2], [0, 3]):
                    for k in range(len(tracer_interpolation_points)):
                        if tracer_interpolation_points[k, index] < value_min:
                            tracer_interpolation_points[k, index] = value_min
                for value_max, index in ([np.where(LSM.lsm > 0)[1].max(), 2], [LSM.z_dim - 1, 3]):
                    for k in range(len(tracer_interpolation_points)):
                        if tracer_interpolation_points[k, index] > value_max:
                            tracer_interpolation_points[k, index] = value_max
            
            interpolation_points.append(tracer_interpolation_points)

        ## load function
        def load_trajectory_function(trajectory_path, tracer_index):
            tracer_trajectory = ndop.model.data.load_trajectories_to_map_index_array(trajectory_path, tracer_index=tracer_index)
            interpolated_values_for_tracer = self._interpolate(tracer_trajectory, interpolation_points[tracer_index])
            return interpolated_values_for_tracer
            

        return load_trajectory_function



    def _interpolate(self, data, interpolation_points, use_cache=False):
        from .constants import MODEL_INTERPOLATOR_FILE, MODEL_INTERPOLATOR_AMOUNT_OF_WRAP_AROUND, MODEL_INTERPOLATOR_NUMBER_OF_LINEAR_INTERPOLATOR, MODEL_INTERPOLATOR_TOTAL_OVERLAPPING_OF_LINEAR_INTERPOLATOR, METOS_DIM

        data_points = data[:,:-1]
        data_values = data[:,-1]
        interpolator_file = MODEL_INTERPOLATOR_FILE

        ## try to get cached interpolator
        interpolator = self._interpolator_cached
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
                interpolator = util.math.interpolate.Periodic_Interpolator(data_points=data_points, data_values=data_values, point_range_size=METOS_DIM, scaling_values=(METOS_DIM[1]/METOS_DIM[0], None, None, None), wrap_around_amount=MODEL_INTERPOLATOR_AMOUNT_OF_WRAP_AROUND, number_of_linear_interpolators=MODEL_INTERPOLATOR_NUMBER_OF_LINEAR_INTERPOLATOR, total_overlapping_linear_interpolators=MODEL_INTERPOLATOR_TOTAL_OVERLAPPING_OF_LINEAR_INTERPOLATOR)
                logger.debug('Returning new created interpolator.')

            self._interpolator_cached = interpolator

        ## interpolate
        interpolated_values = interpolator.interpolate(interpolation_points)

        ## save interpolate if cache used
        if use_cache and not os.path.exists(interpolator_file):
            interpolator.save(interpolator_file)

        ## return interpolated values
        assert not np.any(np.isnan(interpolated_values))
#         assert np.all(interpolator.data_points == data_points)
#         assert np.all(interpolator.data_values == data_values)

        return interpolated_values



    def _f(self, load_trajectory_function, parameters, spinup_options=None, time_step=1):
        from .constants import MODEL_START_FROM_CLOSEST_PARAMETER_SET

        parameters = np.asanyarray(parameters)
        parameter_set_dir = self.get_parameter_set_dir(time_step, parameters, create=True)
        spinup_run_dir = self.get_spinup_run_dir(parameter_set_dir, spinup_options, start_from_closest_parameters=MODEL_START_FROM_CLOSEST_PARAMETER_SET)
        f = self._get_trajectory(load_trajectory_function, spinup_run_dir, parameters)

        assert f is not None
        return f


    def f_boxes(self, parameters, time_dim_desired, spinup_options=None, time_step=1):
        logger.debug('Calculating all f values for parameters {} with time dimension {} and spinup options {}.'.format(parameters, time_dim_desired, spinup_options))

        f = self._f(self._get_load_trajectory_function_for_all(time_dim_desired), parameters, spinup_options, time_step=time_step)

        assert len(f) == 2
        return f


    def f_points(self, parameters, points, spinup_options=None, time_step=1):
        logger.debug('Calculating f values for parameters {} at {} points and spinup options {}.'.format(parameters, tuple(map(len, points)), spinup_options))

        if len(points) != 2:
            raise ValueError('Points have to be a sequence of 2 point arrays. But its length is {}.'.format(len(points)))

        f = self._f(self._get_load_trajectory_function_for_points(points), parameters, spinup_options, time_step=time_step)

        assert len(f) == 2
        assert (not np.any(np.isnan(f[0]))) and (not np.any(np.isnan(f[1])))
        return f




    def _df(self, load_trajectory_function, parameters, spinup_options=None, time_step=1, accuracy_order=2):
        from .constants import MODEL_OUTPUT_DIR, MODEL_DERIVATIVE_DIRNAME, MODEL_SPINUP_DIRNAME, MODEL_PARTIAL_DERIVATIVE_DIRNAME, METOS_TRACER_DIM, MODEL_START_FROM_CLOSEST_PARAMETER_SET, MODEL_PARAMETER_TYPICAL

        MODEL_DERIVATIVE_SPINUP_YEARS = self.df_options['years']
        MODEL_DERIVATIVE_STEP_SIZE = self.df_options['step_size']

        parameters = np.asanyarray(parameters)

        ## chose h factors
        if accuracy_order == 1:
            h_factors = (1,)
        elif accuracy_order == 2:
            h_factors = (1, -1)
        else:
            raise ValueError('Accuracy order {} not supported.'.format(accuracy_order))

        ## search directories
        parameter_set_dir = self.get_parameter_set_dir(time_step, parameters, create=True)
        derivative_dir = os.path.join(parameter_set_dir, MODEL_DERIVATIVE_DIRNAME.format(MODEL_DERIVATIVE_STEP_SIZE))

        ## get spinup run
        years, tolerance, combination = self.all_spinup_options(spinup_options)
        spinup_run_dir = self.get_spinup_run_dir(parameter_set_dir, {'years':years - MODEL_DERIVATIVE_SPINUP_YEARS, 'tolerance':tolerance, 'combination':combination}, start_from_closest_parameters=MODEL_START_FROM_CLOSEST_PARAMETER_SET)
        spinup_run_years = self.get_total_years(spinup_run_dir)

        ## get f if accuracy_order is 1
        if accuracy_order == 1:
            previous_spinup_run_dir = self.get_previous_run_dir(spinup_run_dir)
            previous_spinup_run_years = self.get_total_years(previous_spinup_run_dir)
            if previous_spinup_run_years == spinup_run_years - MODEL_DERIVATIVE_SPINUP_YEARS:
                spinup_run_dir = previous_spinup_run_dir
                spinup_run_years = previous_spinup_run_years

            f = self._f(load_trajectory_function, parameters, {'years':spinup_run_years + MODEL_DERIVATIVE_SPINUP_YEARS, 'tolerance':0, 'combination':'or'}, time_step=time_step)

        ## init values
        parameters_len = len(parameters)
        h_factors_len = len(h_factors)

#         eps = np.spacing(1)
#         eta = np.sqrt(eps) # square root of accuracy of F
#         eta = 10**(-7) # square root of accuracy of F
        h = np.empty((parameters_len, h_factors_len))
        parameters_for_derivative = np.empty((parameters_len, h_factors_len, parameters_len))

        parameters_lower_bound = self.parameters_lower_bound
        parameters_upper_bound = self.parameters_upper_bound

        job_setup = self.get_job_setup('derivative')
        partial_derivative_run_dirs = np.empty([parameters_len, h_factors_len], dtype=object)

        ## start derivative runs
        for parameter_index in range(parameters_len):
            h_i = MODEL_PARAMETER_TYPICAL[parameter_index] * MODEL_DERIVATIVE_STEP_SIZE

            for h_factor_index in range(h_factors_len):

                ## prepare parameters for derivative
                parameters_for_derivative[parameter_index, h_factor_index] = np.copy(parameters)
                h[parameter_index, h_factor_index] = h_factors[h_factor_index] * h_i
                parameters_for_derivative[parameter_index, h_factor_index, parameter_index] += h[parameter_index, h_factor_index]

                ## consider bounds
                violates_lower_bound = parameters_for_derivative[parameter_index, h_factor_index, parameter_index] < parameters_lower_bound[parameter_index]
                violates_upper_bound = parameters_for_derivative[parameter_index, h_factor_index, parameter_index] > parameters_upper_bound[parameter_index]

                if accuracy_order == 1:
                    if violates_lower_bound or violates_upper_bound:
                        h[parameter_index, h_factor_index] *= -1
                        parameters_for_derivative[parameter_index, h_factor_index, parameter_index] = parameters[parameter_index] + h[parameter_index, h_factor_index]
                else:
                    if violates_lower_bound:
                        parameters_for_derivative[parameter_index, h_factor_index, parameter_index] = parameters_lower_bound[parameter_index]
                    elif violates_upper_bound:
                        parameters_for_derivative[parameter_index, h_factor_index, parameter_index] = parameters_upper_bound[parameter_index]

                ## calculate h   (improvement of accuracy of h)
                h[parameter_index, h_factor_index] = parameters_for_derivative[parameter_index, h_factor_index, parameter_index] - parameters[parameter_index]

                logger.debug('Calculating finite differences approximation for parameter index {} with h value {}.'.format(parameter_index, h[parameter_index, h_factor_index]))

                ## get run dir
                h_factor = int(np.sign(h[parameter_index, h_factor_index]))
                partial_derivative_dirname = MODEL_PARTIAL_DERIVATIVE_DIRNAME.format(parameter_index, h_factor)
                partial_derivative_dir = os.path.join(derivative_dir, partial_derivative_dirname)
                partial_derivative_run_dir = self.get_last_run_dir(partial_derivative_dir)

                ## make new run if run not matching
                if not self.is_run_matching_options(partial_derivative_run_dir, {'years':MODEL_DERIVATIVE_SPINUP_YEARS, 'tolerance':0, 'combination':'or'}):
                    util.io.fs.remove_recursively(partial_derivative_dir, force=True, exclude_dir=True)

                    ## if no job setup available, get best job setup
                    if job_setup['nodes_setup'] is None:
                        try:
                            nodes_max = job_setup['nodes_max']
                        except KeyError:
                            nodes_max = None
                        job_setup['nodes_setup'] = ndop.model.job.Metos3D_Job.best_nodes_setup(years, nodes_max=nodes_max)

                    ## start job
                    partial_derivative_run_dir = self.make_run(partial_derivative_dir, parameters_for_derivative[parameter_index, h_factor_index], MODEL_DERIVATIVE_SPINUP_YEARS, 0, time_step, job_setup, tracer_input_path=spinup_run_dir, wait_until_finished=False)

                partial_derivative_run_dirs[parameter_index, h_factor_index] = partial_derivative_run_dir


        ## make trajectories and calculate df
        df = [None] * METOS_TRACER_DIM

        for parameter_index in range(parameters_len):
            for h_factor_index in range(h_factors_len):
                ## wait partial derivative run to finish
                partial_derivative_run_dir = partial_derivative_run_dirs[parameter_index, h_factor_index]
                self.wait_until_job_finished(partial_derivative_run_dir)

                ## get trajectory and add to df
                trajectory = self._get_trajectory(load_trajectory_function, partial_derivative_run_dir, parameters_for_derivative[parameter_index, h_factor_index])
                for tracer_index in range(METOS_TRACER_DIM):
                    if df[tracer_index] is None:
                        df[tracer_index] = np.zeros((parameters_len,) + trajectory[tracer_index].shape)
                    df[tracer_index][parameter_index] += (-1)**h_factor_index * trajectory[tracer_index]

            ## calculate df
            for tracer_index in range(METOS_TRACER_DIM):
                if accuracy_order == 1:
                    df[tracer_index][parameter_index] -= f[tracer_index]
                    df[tracer_index][parameter_index] /= h[parameter_index]
                else:
                    df[tracer_index][parameter_index] /= np.sum(np.abs(h[parameter_index]))

        return df


    def df_boxes(self, parameters, time_dim_desired, spinup_options=None, time_step=1, accuracy_order=2):
        logger.debug('Calculating all df values for parameters {} with time dimension {}, spinup options {} and accuracy order {}.'.format(parameters, time_dim_desired, spinup_options, accuracy_order))

        df = self._df(self._get_load_trajectory_function_for_all(time_dim_desired=time_dim_desired), parameters, spinup_options, time_step=time_step, accuracy_order=accuracy_order)

        assert len(df) == 2
        return df


    def df_points(self, parameters, points, spinup_options=None, time_step=1, accuracy_order=2):
        logger.debug('Calculating df values for parameters {} at {} points with spinup options {} and accuracy order {}.'.format(parameters, tuple(map(len, points)), spinup_options, accuracy_order))

        df = self._df(self._get_load_trajectory_function_for_points(points), parameters, spinup_options, time_step=time_step, accuracy_order=accuracy_order)

        assert len(df) == 2
        assert (not np.any(np.isnan(df[0]))) and (not np.any(np.isnan(df[1])))
        return df
