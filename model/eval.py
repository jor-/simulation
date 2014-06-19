import os
import warnings
import tempfile
import stat
import time
import numpy as np
import multiprocessing
import multiprocessing.pool

import logging
logger = logging.getLogger(__name__)

import util.io
import util.pattern
import util.math.interpolate
import measurements.util.interpolate

import ndop.model.data
from ndop.model.job import Metos3D_Job


class Model():
    
    # TODO run setup (years, tolerance=0, combination='or', time_step=1, df_accuracy_order=2) zusammenfassen
#     def __init__(self, job_name_prefix='', job_nodes_setup=None, job_nodes_max_file=None):
    def __init__(self, job_setup=None):
        from .constants import MODEL_PARAMETER_LOWER_BOUND, MODEL_PARAMETER_UPPER_BOUND, MODEL_OUTPUT_DIR, METOS_TRACER_DIM
        
        logger.debug('Model initiated with job_setup {}.'.format(job_setup))
        
#         self.job_name_prefix = job_name_prefix
#         self.job_nodes_setup = job_nodes_setup
#         self.job_nodes_max_file = job_nodes_max_file
        self.job_setup = job_setup
        
        self.parameters_lower_bound = MODEL_PARAMETER_LOWER_BOUND
        self.parameters_upper_bound = MODEL_PARAMETER_UPPER_BOUND
        self.model_output_dir = MODEL_OUTPUT_DIR
        
        self._interpolator_cached = None
    
    
    @property
    def land_sea_mask(self):
        """The land-sea-mask from metos3d as numpy array."""
        
        logger.debug('Getting land-sea-mask.')
        
        ## return land-sea-mask if loaded
        try:
            land_sea_mask = self.__land_sea_mask
            logger.debug('Returning cached land-sea-mask.')
        ## otherwise load land-sea-mask and return it
        except AttributeError:
            land_sea_mask = ndop.model.data.load_land_sea_mask()
            self.__land_sea_mask = land_sea_mask
        
        logger.debug('Got land-sea-mask.')
        
        return land_sea_mask
    
    
    
    def check_if_parameters_in_bounds(self, parameters):
        if any(parameters < self.parameters_lower_bound):
            indices = np.where(parameters < self.parameters_lower_bound)
            string = ''
            for i in indices:
                string += str(parameters[i]) + '<' + str(self.parameters_lower_bound[i])
                if i < len(indices) - 1:
                    string += ', '
            raise ValueError('Some parameters are lower than the lower bound: ' + string)
        
        if any(parameters > self.parameters_upper_bound):
            indices = np.where(parameters > self.parameters_upper_bound)
            string = ''
            for i in indices:
                string += str(parameters[i]) + '>' + str(self.parameters_upper_bound[i])
                if i < len(indices) - 1:
                    string += ', '
            raise ValueError('Some parameters are upper than the upper bound: ' + string)
    
    
    
    def check_combination_value(self, value):
        valid_values = ('and', 'or')
        if not value in valid_values:
            raise ValueError('The combination value {} is not valid. Only the values {} are supported.'.format(value, valid_values))
    
    
    
    
#     def search_or_make_time_step_dir(self, search_path, time_step):
#         from .constants import MODEL_TIME_STEP_DIRNAME
#         
#         self.print_debug_inc(('Searching for directory for time step size "', time_step, '" in "', search_path, '".'))
#         
#         time_step_dirname = util.pattern.replace_int_pattern(MODEL_TIME_STEP_DIRNAME, time_step)
#         time_step_dir = os.path.join(search_path, time_step_dirname)
#         
#         try:
#             os.makedirs(time_step_dir)
#             self.print_debug_dec(('Directory "', time_step_dir, '" created.'))
#         except OSError:
#             self.print_debug_dec(('Directory "', time_step_dir, '" found.'))
#         
#         return time_step_dir
    
    
    
    
    
#     def is_matching_parameter_set(self, parameters, parameter_set_dir):
#         from .constants import MODEL_PARAMETERS_MAX_DIFF
#         
#         parameters_diff = self.get_parameters_diff(parameters, parameter_set_dir)
#         is_matching = parameters_diff <= MODEL_PARAMETERS_MAX_DIFF
#         
#         return is_matching
#     
#     
#     
#     def search_nearest_parameter_set_dir(self, parameters, search_path):
#         self.print_debug_inc(('Searching for directory for parameters as close as possible to "', parameters, '" in "', search_path, '".'))
#         
#         ## check input
#         self.check_if_parameters_in_bounds(parameters)
#         
#         
#         ## search for directories with matching parameters
#         parameter_set_dir = None
#         parameter_set_number = 0
#         parameters_diff = float('inf')
#         
#         if os.path.exists(search_path):
#             parameter_set_dirs = util.io.get_dirs(search_path)
#             number_of_parameter_set_dirs = len(parameter_set_dirs)
#             while parameters_diff > 0 and parameter_set_number < number_of_parameter_set_dirs:
#                 current_parameter_set_dir = parameter_set_dirs[parameter_set_number]
#                 try:
#                     current_parameters_diff = self.get_parameters_diff(parameters, current_parameter_set_dir)
#                 except (OSError, IOError):
#                     warnings.warn('Could not read the parameters file "' + current_parameters_file + '"!')
#                     current_parameters_diff = float('inf')
#                 
#                 if current_parameters_diff < parameters_diff:
#                     parameter_set_dir = current_parameter_set_dir
#                     parameters_diff = current_parameters_diff
#                 
#                 parameter_set_number += 1
#         
#         ## return parameter_set_dir with min diff
#         if parameter_set_dir is not None:
#             self.print_debug_dec(('Directory closest to parameters at "', parameter_set_dir, '" found.'))
#         else:
#             self.print_debug_dec('No directory closest to parameters found.')
#         
#         return parameter_set_dir
#     
#     
#     
#     def make_new_parameter_set_dir(self, path, parameters):
#         from .constants import MODEL_PARAMETERS_SET_DIRNAME, MODEL_PARAMETERS_FILENAME, MODEL_PARAMETERS_FORMAT_STRING
#         
#         self.print_debug_inc(('Creating new parameter set directory in "', path, '".'))
#         
#         parameter_set_number = 0
#         parameter_set_dir = None
#         while parameter_set_dir is None:
#             parameter_set_dirname = util.pattern.replace_int_pattern(MODEL_PARAMETERS_SET_DIRNAME, parameter_set_number)
#             parameter_set_dir_candidate = os.path.join(path, parameter_set_dirname)
#             if os.path.isdir(parameter_set_dir_candidate):
#                 parameter_set_number +=  1
#             else:
#                 os.makedirs(parameter_set_dir_candidate)
#                 parameter_set_dir = parameter_set_dir_candidate
#         
#         parameters_file = os.path.join(parameter_set_dir, MODEL_PARAMETERS_FILENAME)
#         np.savetxt(parameters_file, parameters, fmt=MODEL_PARAMETERS_FORMAT_STRING)
#         os.chmod(parameters_file, stat.S_IRUSR)
#         self.print_debug_dec(('New parameter set directory "', parameter_set_dir, '" created.'))
#         
#         return parameter_set_dir
#         
#         
#     def search_or_make_parameter_set_dir(self, search_path, parameters):
#         logger.debug('Searching for directory for parameters {} in {}.'.format(parameters, search_path))
#         
#         ## check input
#         self.check_if_parameters_in_bounds(parameters)
#         
#         ## search for directories with matching parameters
#         parameter_set_dir = None
# #         parameter_set_number = 0
#         
#         ## if search path exists, search for matching parameter set 
#         if os.path.exists(search_path):
#             from .constants import MODEL_PARAMETERS_MAX_DIFF
#             
# #             closest_parameter_set_dir = self.search_nearest_parameter_set_dir(parameters, search_path)
#             
#             logger.debug('Searching for directory for parameters as close as possible to {} in {}.'.format(parameters, search_path))
#             
#             ## search for parameter set directories nearby parameters
#             closest_parameter_set_dir = None
#             closest_parameters_diff = float('inf')
#             parameter_set_number = 0
#             
#             parameter_set_dirs = util.io.get_dirs(search_path)
#             number_of_parameter_set_dirs = len(parameter_set_dirs)
#             while closest_parameters_diff > 0 and parameter_set_number < number_of_parameter_set_dirs:
#                 current_parameter_set_dir = parameter_set_dirs[parameter_set_number]
#                 try:
#                     current_parameters_diff = self.get_parameters_diff(parameters, current_parameter_set_dir)
#                 except (OSError, IOError):
#                     warnings.warn('Could not read the parameters file {}!'.format(current_parameters_file))
#                     current_parameters_diff = float('inf')
#                 
#                 if current_parameters_diff < closest_parameters_diff:
#                     closest_parameter_set_dir = current_parameter_set_dir
#                     closest_parameters_diff = current_parameters_diff
#                 
#                 parameter_set_number += 1
#             
#             ## return closest parameter set dir if it matches the parameters
# #             if self.is_matching_parameter_set(parameters, closest_parameter_set_dir):
#             if closest_parameters_diff <= MODEL_PARAMETERS_MAX_DIFF:
#                 parameter_set_dir = closest_parameter_set_dir
#                 logger.debug('Existing directory for parameters found at {}.'.format(parameter_set_dir))
#         else:
#             os.makedirs(search_path)
#             number_of_parameter_set_dirs = 0
#         
#         
#         ## make new model_output if the parameters are not matching
#         if parameter_set_dir is None:
#             from .constants import MODEL_PARAMETERS_SET_DIRNAME, MODEL_PARAMETERS_FILENAME, MODEL_PARAMETERS_FORMAT_STRING
#             
# #             parameter_set_number = 0
# #             parameter_set_dir = None
# #             while parameter_set_dir is None:
# #                 parameter_set_dirname = util.pattern.replace_int_pattern(MODEL_PARAMETERS_SET_DIRNAME, parameter_set_number)
# #                 parameter_set_dir_candidate = os.path.join(search_path, parameter_set_dirname)
# #                 if os.path.isdir(parameter_set_dir_candidate):
# #                     parameter_set_number +=  1
# #                 else:
# #                     os.makedirs(parameter_set_dir_candidate)
# #                     parameter_set_dir = parameter_set_dir_candidate
#             
#             parameter_set_dirname = util.pattern.replace_int_pattern(MODEL_PARAMETERS_SET_DIRNAME, number_of_parameter_set_dirs)
#             parameter_set_dir = os.path.join(search_path, parameter_set_dirname)
#             
#             parameters_file = os.path.join(parameter_set_dir, MODEL_PARAMETERS_FILENAME)
#             np.savetxt(parameters_file, parameters, fmt=MODEL_PARAMETERS_FORMAT_STRING)
#             os.chmod(parameters_file, stat.S_IRUSR)
#             
# #             parameter_set_dir = self.make_new_parameter_set_dir(search_path, parameters)
#             logger.debug('Directory for parameters {} created in {}.'.format(parameters, parameter_set_dir))
#         else:
#             logger.debug('Matching directory for parameters at {} found.'.format(parameter_set_dir))
#         
#         return parameter_set_dir
    
    
    
    
    def get_parameters_diff(self, parameters, parameter_set_dir):
        from .constants import MODEL_PARAMETERS_FILENAME
        
        parameters_file = os.path.join(parameter_set_dir, MODEL_PARAMETERS_FILENAME)
        current_parameters = np.loadtxt(parameters_file)
        
        parameter_diff = np.linalg.norm(parameters - current_parameters)
        
        return parameter_diff
    
    
    
    ## interpolate
    
    def _interpolate(self, tracer_index, data, interpolation_points, use_cache=False):
        from .constants import MODEL_INTERPOLATOR_FILE, MODEL_INTERPOLATOR_AMOUNT_OF_WRAP_AROUND, MODEL_INTERPOLATOR_NUMBER_OF_LINEAR_INTERPOLATOR, MODEL_INTERPOLATOR_TOTAL_OVERLAPPING_OF_LINEAR_INTERPOLATOR
        
        data_points = data[:,:-1]
        data_values = data[:,-1]
#         interpolator_file = MODEL_INTERPOLATOR_FILE.format(tracer_index)
        interpolator_file = MODEL_INTERPOLATOR_FILE
        
        ## try to get cached interpolator
#         interpolator = self._interpolators_cached[tracer_index]
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
                interpolator = measurements.util.interpolate.Time_Periodic_Earth_Interpolater(data_points=data_points, data_values=data_values, t_len=1, wrap_around_amount=MODEL_INTERPOLATOR_AMOUNT_OF_WRAP_AROUND, number_of_linear_interpolators=MODEL_INTERPOLATOR_NUMBER_OF_LINEAR_INTERPOLATOR, total_overlapping_linear_interpolators=MODEL_INTERPOLATOR_TOTAL_OVERLAPPING_OF_LINEAR_INTERPOLATOR)
                logger.debug('Returning new created interpolator.')
            
#             self._interpolators_cached[tracer_index] = interpolator
            self._interpolator_cached = interpolator
        
        ## interpolate
        interpolated_values = interpolator.interpolate(interpolation_points)
        
        ## save interpolate if cache used
        if use_cache and not os.path.exists(interpolator_file):
            interpolator.save(interpolator_file)
        
        ## return interpolated values
#         assert self._interpolators_cached[tracer_index] is interpolator
        assert not np.any(np.isnan(interpolated_values))
#         assert np.all(interpolator.data_points == data_points)
#         assert np.all(interpolator.data_values == data_values)
        
        return interpolated_values
    
    
    
    
    ## access to dirs
    
    def get_time_step_dir(self, time_step, create=True):
        from .constants import MODEL_TIME_STEP_DIRNAME
        
        ## get directory for time step
        model_output_dir = self.model_output_dir
        logger.debug('Searching for directory for time step size {} in {}.'.format(time_step, model_output_dir))
        
#         time_step_dirname = util.pattern.replace_int_pattern(MODEL_TIME_STEP_DIRNAME, time_step)
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
            from .constants import MODEL_PARAMETERS_MAX_DIFF, MODEL_SPINUP_DIRNAME
            
            logger.debug('Searching for directory for parameters as close as possible to {} in {}.'.format(parameters, time_step_dir))
            
            ## search for parameter set directories with nearby parameters
            parameter_set_number = 0
            
            parameter_set_dirs = util.io.get_dirs(time_step_dir)
            number_of_parameter_set_dirs = len(parameter_set_dirs)
            while closest_parameters_diff > 0 and parameter_set_number < number_of_parameter_set_dirs:
                current_parameter_set_dir = parameter_set_dirs[parameter_set_number]
                try:
                    current_parameters_diff = self.get_parameters_diff(parameters, current_parameter_set_dir)
                except (OSError, IOError):
                    warnings.warn('Could not read the parameters file {}!'.format(current_parameters_file))
                    current_parameters_diff = float('inf')
                
                if current_parameters_diff < closest_parameters_diff:
                    better = False
                    if not no_spinup_okay:
                        current_spinup_dir = os.path.join(current_parameter_set_dir, MODEL_SPINUP_DIRNAME)
                        last_run_dir = self.get_last_run_dir(current_spinup_dir, wait_until_finished=False)
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
        from .constants import MODEL_PARAMETERS_MAX_DIFF
        
        ## search for directories with matching parameters
        logger.debug('Searching parameter directory for time_step {} and parameters {}.'.format(time_step, parameters))
        
        parameter_set_dir = self.get_closest_parameter_set_dir(time_step, parameters, no_spinup_okay=True)
        if parameter_set_dir is not None:
            parameters_diff = self.get_parameters_diff(parameters, parameter_set_dir)
            if parameters_diff > MODEL_PARAMETERS_MAX_DIFF:
                parameter_set_dir = None
        
        
        ## make new model_output if the parameters are not matching
        if parameter_set_dir is None:
            if create:
                from .constants import MODEL_PARAMETERS_SET_DIRNAME, MODEL_PARAMETERS_FILENAME, MODEL_PARAMETERS_FORMAT_STRING
                
                time_step_dir = self.get_time_step_dir(time_step, create=True)
                
#                 number_of_parameter_set_dirs = len(util.io.get_dirs(time_step_dir))
#                 parameter_set_dirname = util.pattern.replace_int_pattern(MODEL_PARAMETERS_SET_DIRNAME, number_of_parameter_set_dirs)
                ## search free index
                free_parameter_set_dir_number = len(util.io.get_dirs(time_step_dir))
                free_parameter_set_dir_number = 0
                found = False
                while not found:
                    parameter_set_dirname = MODEL_PARAMETERS_SET_DIRNAME.format(free_parameter_set_dir_number)
                    parameter_set_dir = os.path.join(time_step_dir, parameter_set_dirname)
                    found = not os.path.exists(parameter_set_dir)
                    free_parameter_set_dir_number += 1
                
                ## make dir
                os.makedirs(parameter_set_dir)
                
                parameters_file = os.path.join(parameter_set_dir, MODEL_PARAMETERS_FILENAME)
                np.savetxt(parameters_file, parameters, fmt=MODEL_PARAMETERS_FORMAT_STRING)
                os.chmod(parameters_file, stat.S_IRUSR)
                
                logger.debug('Directory for parameters created in {}.'.format(parameter_set_dir))
            else:
                logger.debug('No matching directory for parameters found. Non created.')
        else:
            logger.debug('Matching directory for parameters found at {}.'.format(parameter_set_dir))
        
        return parameter_set_dir
    
    
    def get_spinup_run_dir(self, parameter_set_dir, years, tolerance, combination, start_from_closest_parameters=False):
        from .constants import MODEL_SPINUP_DIRNAME, MODEL_PARAMETERS_FILENAME, MODEL_SPINUP_MAX_YEARS
        
        self.check_combination_value(combination)
        
        spinup_dir = os.path.join(parameter_set_dir, MODEL_SPINUP_DIRNAME)
        
        logger.debug('Searching for spinup with {} years {} {} tolerance in {}.'.format(years, combination, tolerance, spinup_dir))
        
        last_run_dir = self.get_last_run_dir(spinup_dir)
        
        ## matching spinup found
        if self.is_run_matching_options(last_run_dir, years, tolerance, combination=combination):
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
                with Metos3D_Job(last_run_dir, force_load=True) as job:
                    job.wait_until_finished()
            
            ## make new run
            if combination == 'or':
                run_dir = self.make_run(spinup_dir, parameters, years, tolerance, time_step, tracer_input_path=last_run_dir)
            elif combination == 'and':
                run_dir = self.get_spinup_run_dir(parameter_set_dir, years, 0, 'or', start_from_closest_parameters)
                run_dir = self.get_spinup_run_dir(parameter_set_dir, MODEL_SPINUP_MAX_YEARS, tolerance, 'or', start_from_closest_parameters)
            
            logger.debug('Spinup directory created at {}.'.format(run_dir))
            
#             self.remove_f(parameter_set_dir)
        
        return run_dir
    
    
    
    def make_run(self, output_path, parameters, years, tolerance, time_step, tracer_input_path=None):
        from .constants import MODEL_RUN_DIRNAME, MODEL_RUN_OPTIONS_FILENAME
        
        ## check parameters
        self.check_if_parameters_in_bounds(parameters)
        
        ## get next run index
        util.io.makedirs_if_not_exists(output_path)
        run_dirs = self.get_run_dirs(output_path)
        next_run_index = len(run_dirs)
        
        ## create run dir
#         run_dirname = util.pattern.replace_int_pattern(MODEL_RUN_DIRNAME, next_run_index)
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
            
        self.run_job(parameters, output_path=run_dir, tracer_input_path=tracer_input_path, years=years-last_years, tolerance=tolerance, time_step=time_step, wait_pause_seconds=150)
        
        return run_dir
    
    
    
    def run_job(self, model_parameters, output_path, years, tolerance, time_step, write_trajectory=False, tracer_input_path=None, wait_pause_seconds=10, make_read_only=True):
        logger.debug('Running job with years {} tolerance {} time_step {} tracer_input_path {}'.format(years, tolerance, time_step, tracer_input_path))
        assert years >= 0
        assert tolerance >= 0
        assert time_step >= 1
        
        ## check parameters
        self.check_if_parameters_in_bounds(model_parameters)
        
        ## load max nodes
#         nodes_max_file = self.job_nodes_max_file
#         if nodes_max_file is not None:
#             try:
#                 nodes_max = np.loadtxt(nodes_max_file)
#             except (OSError, IOError):
#                 warnings.warn('The nodes_max_file {} could not been read.'.format(nodes_max_file))
#                 nodes_max = None
#         else:
#             nodes_max = None
#         nodes_max = self.job_nodes_max_file
#         
#         nodes_setup = self.job_nodes_setup
        
        ## execute job
        job_setup = self.job_setup
        if job_setup is not None:
            job_setup = job_setup.copy()
            if years <= 250:
                job_setup['nodes_setup'] = None
#             nodes_setup = None
        with Metos3D_Job(output_path, wait_pause_seconds) as job:
#             job.init(model_parameters, years=years, tolerance=tolerance, time_step=time_step, write_trajectory=write_trajectory, tracer_input_path=tracer_input_path, nodes_setup=nodes_setup, nodes_max=nodes_max, job_name_prefix=self.job_name_prefix)
            job.init(model_parameters, years=years, tolerance=tolerance, time_step=time_step, write_trajectory=write_trajectory, tracer_input_path=tracer_input_path, job_setup=job_setup)
            job.start()
            if make_read_only:
                job.make_read_only()
            job.wait_until_finished()
        
        ## change access mode of files
        if make_read_only:
            util.io.make_read_only_recursively(output_path, exclude_dir=True)
    
    
    
    
#     def is_run_matching_options(self, run_dir, years, tolerance, time_step):
#         if run_dir is not None:
#             run_years = self.get_total_years(run_dir)
#             run_tolerance = self.get_real_tolerance(run_dir)
#             run_time_step = self.get_time_step(run_dir)
#             
#             
#             is_matching = time_step >= run_time_step and (years <= run_years or tolerance >= run_tolerance)
#         else:
#             is_matching = False
#         
#         return is_matching
    
    ## access to runs
    def is_run_matching_options(self, run_dir, years, tolerance, combination='or'):
        from .constants import MODEL_SPINUP_MAX_YEARS
        
        if run_dir is not None:
            run_years = self.get_total_years(run_dir)
            run_tolerance = self.get_real_tolerance(run_dir)
            
            self.check_combination_value(combination)
            if combination == 'and':
                is_matching = (run_years >= years and (run_tolerance <= tolerance or run_years >= MODEL_SPINUP_MAX_YEARS))
                if is_matching and run_tolerance > tolerance:
                    warnings.warn('The run {} does not match the desired tolerance {}, but the max spinup years {} are reached.'.format(run_dir, tolerance, MODEL_SPINUP_MAX_YEARS))
            elif combination == 'or':
                is_matching = (run_years >= years or run_tolerance <= tolerance)
            else:
                raise ValueError('Combination "{}" unknown.'.format(combination))
                
        else:
            is_matching = False
        
        return is_matching
    
    
    
    def get_run_dirs(self, search_path):
        from .constants import MODEL_RUN_DIRNAME
        
        run_dir_condition = lambda file: os.path.isdir(file) and util.pattern.is_matching(os.path.basename(file), MODEL_RUN_DIRNAME)
        try:
            run_dirs = util.io.filter_files(search_path, run_dir_condition)
        except (OSError, IOError) as exception:
            warnings.warn('It could not been searched in the search path "' + search_path + '": ' + str(exception))
            run_dirs = []
        
        return run_dirs
    
    
    
    def get_last_run_dir(self, search_path, wait_until_finished=True):
        logger.debug('Searching for last run in {} with wait_until_finished {}.'.format(search_path, wait_until_finished))
        
        run_dirs = self.get_run_dirs(search_path)
        
        last_run_dir = None
        last_run_index =  len(run_dirs) - 1
        
        while last_run_dir is None and last_run_index >= 0:
            last_run_dir = run_dirs[last_run_index]
            logger.debug('Searching in {}.'.format(last_run_dir))
            
            # load last job finished
            if last_run_dir is not None:
                try: 
                    job = Metos3D_Job(last_run_dir, force_load=True)
                except (OSError, IOError) as exception:
                    warnings.warn('Could not read the job options file from "' + last_run_dir + '": ' + str(exception))
                    job = None
                    last_run_dir = None
                
                # check if job is finished
                if wait_until_finished and last_run_dir is not None:
                    try:
                        job.wait_until_finished()
                    except (OSError, IOError) as exception:
                        warnings.warn('Could not check if job ' + job.id + ' is finished: ' + str(exception))
                        last_run_dir = None
                
                if job is not None:
                    job.close()
            
            last_run_index -= 1
        
        logger.debug('Run {} found.'.format(last_run_dir))
        
        return last_run_dir
    
    
    
    def get_previous_run_dir(self, run_dir):
        from .constants import MODEL_RUN_DIRNAME
        
        (spinup_dir, run_dirname) = os.path.split(run_dir)
        #run_index = int(re.findall('\d+', run_dirname)[0])
        run_index = util.pattern.get_int_in_string(run_dirname)
        if run_index > 0:
#             previous_run_dirname = util.pattern.replace_int_pattern(MODEL_RUN_DIRNAME, run_index - 1)
            previous_run_dirname = MODEL_RUN_DIRNAME.format(run_index - 1)
            previous_run_dir = os.path.join(spinup_dir, previous_run_dirname)
        else:
            previous_run_dir = None
        
        return previous_run_dir
    
    
    
    ##  access run properties
    def get_total_years(self, run_dir):
        total_years = 0
        
        while run_dir is not None:
            with Metos3D_Job(run_dir, force_load=True) as job:
                years = job.last_year
            total_years += years
            run_dir = self.get_previous_run_dir(run_dir)
        
        return total_years
    
    
    
    def get_real_tolerance(self, run_dir):
        with Metos3D_Job(run_dir, force_load=True) as job:
            tolerance = job.last_tolerance     
        
        return tolerance
    
    
    
    def get_time_step(self, run_dir):
        from .constants import MODEL_RUN_OPTIONS_FILENAME  
        
        with Metos3D_Job(run_dir, force_load=True) as job:
            time_step = job.time_step
        
        return time_step
    
    
    
    def get_tracer_input_dir(self, run_dir):
        with Metos3D_Job(run_dir, force_load=True) as job:
            tracer_input_dir = job.tracer_input_path
        
        return tracer_input_dir
    
    
    
    
    
    ## access to model values
    
    def _get_trajectory(self, load_trajectory_function, run_dir, parameters):
        from .constants import MODEL_TMP_DIR, METOS_TRACER_DIM
        
        assert callable(load_trajectory_function)
        
        run_time_step = self.get_time_step(run_dir)
        trajectory_values = ()
        
        ## create trajectory
        if MODEL_TMP_DIR is not None:
            tmp_dir = MODEL_TMP_DIR
        else:
            tmp_dir = run_dir
        with tempfile.TemporaryDirectory(dir=tmp_dir, prefix='trajectory_tmp_') as trajectory_dir:
            self.run_job(parameters, output_path=trajectory_dir, tracer_input_path=run_dir, write_trajectory=True, years=1, tolerance=0, time_step=run_time_step, wait_pause_seconds=10, make_read_only=False)
            
            trajectory_output_dir = os.path.join(trajectory_dir, 'trajectory')
            
            ## for each tracer read trajectory
            for tracer_index in range(METOS_TRACER_DIM):
                tracer_trajectory_values = load_trajectory_function(trajectory_output_dir, tracer_index)
                trajectory_values += (tracer_trajectory_values,)
        
        assert len(trajectory_values) == METOS_TRACER_DIM
        
        return trajectory_values
    
    
    def _get_load_trajectory_function_for_all(self, time_dim_desired):
        
#         def load_trajectory_function(trajectory_path, tracer_index):
#             retry_count = 5
#             tracer_trajectory = None
#             while retry_count >= 1 and tracer_trajectory is None:
#                 try:
#                     trajectory = ndop.model.data.load_trajectories_to_index_array(trajectory_path, tracer_index=tracer_index, land_sea_mask=self.land_sea_mask, time_dim_desired=time_dim_desired)
#                 except FileNotFoundError as e:
#                     if retry_count > 0:
#                         warnings.warn('PETSc vectors in {} not found. Waiting.'.format(trajectory_path))
#                         retry_count -= 1
#                         time.sleep(30)
#                     else:
#                         raise e
#             return trajectory
        
        
        load_trajectory_function = lambda trajectory_path, tracer_index : ndop.model.data.load_trajectories_to_index_array(trajectory_path, tracer_index, land_sea_mask=self.land_sea_mask, time_dim_desired=time_dim_desired)
        return load_trajectory_function
    
    
    
    def _get_load_trajectory_function_for_points(self, points):
        ## discard year
        interpolation_points = []
        for tracer_points in points:
            tracer_interpolation_points = np.array(tracer_points, copy=True)
            tracer_interpolation_points[:, 0] = tracer_interpolation_points[:, 0] % 1
            interpolation_points.append(tracer_interpolation_points)
        
        ## load function
        def load_trajectory_function(trajectory_path, tracer_index):
            tracer_trajectory = ndop.model.data.load_trajectories_to_point_array(trajectory_path, tracer_index=tracer_index, land_sea_mask=self.land_sea_mask)
            interpolated_values_for_tracer = self._interpolate(tracer_index, tracer_trajectory, interpolation_points[tracer_index])
            return interpolated_values_for_tracer
#         def load_trajectory_function(trajectory_path, tracer_index):
#             retry_count = 5
#             tracer_trajectory = None
#             logger.info('Okay?={}'.format(retry_count >= 1 and tracer_trajectory is None))
#             while retry_count >= 1 and tracer_trajectory is None:
#                 try:
#                     tracer_trajectory = ndop.model.data.load_trajectories_to_point_array(trajectory_path, tracer_index=tracer_index, land_sea_mask=self.land_sea_mask)
#                     logger.info('tracer_trajectory={}'.format(tracer_trajectory))
#                 except FileNotFoundError as e:
#                     logger.info('error retry_count={}'.format(retry_count))
#                     if retry_count > 0:
#                         warnings.warn('PETSc vectors in {} not found. Waiting.'.format(trajectory_path))
#                         retry_count -= 1
#                         time.sleep(30)
#                     else:
#                         raise e
#             interpolated_values_for_tracer = self._interpolate(tracer_index, tracer_trajectory, interpolation_points[tracer_index])
#             return interpolated_values_for_tracer
        return load_trajectory_function
    
    
    
    def _f(self, load_trajectory_function, parameters, years, tolerance=0, combination='or', time_step=1):
        from .constants import MODEL_START_FROM_CLOSEST_PARAMETER_SET
        
        logger.debug('Calculating f values with {} time_step, {} years, {} tolerance and combination "{}" as spinup  options.'.format(time_step, years, tolerance, combination))
        
        parameter_set_dir = self.get_parameter_set_dir(time_step, parameters, create=True)
        spinup_run_dir = self.get_spinup_run_dir(parameter_set_dir, years, tolerance, combination=combination, start_from_closest_parameters=MODEL_START_FROM_CLOSEST_PARAMETER_SET)
        f = self._get_trajectory(load_trajectory_function, spinup_run_dir, parameters)
        
        return f
    
    
    def f_all(self, parameters, time_dim_desired, years, tolerance=0, combination='or', time_step=1):
        logger.debug('Calculating all f values for parameters {} with time dimension {}.'.format(parameters, time_dim_desired))
        
        f = self._f(self._get_load_trajectory_function_for_all(time_dim_desired), parameters, years, tolerance=tolerance, combination=combination, time_step=time_step)
        
        assert len(f) == 2
        return f
    
    
    def f_points(self, parameters, points, years, tolerance=0, combination='or', time_step=1):
        logger.debug('Calculating f values for parameters {} at {} points.'.format(parameters, tuple(map(len, points))))
        
        f = self._f(self._get_load_trajectory_function_for_points(points), parameters, years, tolerance=tolerance, combination=combination, time_step=time_step)
        
        assert len(f) == 2
        assert (not np.any(np.isnan(f[0]))) and (not np.any(np.isnan(f[1])))
        return f
    
    
    
    
    def _df(self, load_trajectory_function, parameters, years, tolerance=0, combination='or', time_step=1, accuracy_order=2):
        from .constants import MODEL_OUTPUT_DIR, MODEL_DERIVATIVE_DIRNAME, MODEL_SPINUP_DIRNAME, MODEL_PARTIAL_DERIVATIVE_DIRNAME, MODEL_DERIVATIVE_SPINUP_YEARS, METOS_TRACER_DIM, MODEL_START_FROM_CLOSEST_PARAMETER_SET
        
        logger.debug('Calculating df values with {} time_step, {} years, {} tolerance and combination "{}" as spinup  options.'.format(time_step, years, tolerance, combination))
        
        ## chose h factors
        if accuracy_order == 1:
            h_factors = (1,)
        elif accuracy_order == 2:
            h_factors = (1, -1)
        else:
            raise ValueError('Accuracy order {} not supported.'.format(accuracy_order))
        
        ## search directories
        parameter_set_dir = self.get_parameter_set_dir(time_step, parameters, create=True)
        derivative_dir = os.path.join(parameter_set_dir, MODEL_DERIVATIVE_DIRNAME)
        
        ## get spinup run
        spinup_run_dir = self.get_spinup_run_dir(parameter_set_dir, years - MODEL_DERIVATIVE_SPINUP_YEARS, tolerance, combination, start_from_closest_parameters=MODEL_START_FROM_CLOSEST_PARAMETER_SET)
        spinup_run_years = self.get_total_years(spinup_run_dir)
        
        ## get f if accuracy_order is 1
        if accuracy_order == 1:
            previous_spinup_run_dir = self.get_previous_run_dir(spinup_run_dir)
            previous_spinup_run_years = self.get_total_years(previous_spinup_run_dir)
            if previous_spinup_run_years == spinup_run_years - MODEL_DERIVATIVE_SPINUP_YEARS:
                spinup_run_dir = previous_spinup_run_dir
                spinup_run_years = previous_spinup_run_years
            
            f = self._f(load_trajectory_function, parameters, spinup_run_years + MODEL_DERIVATIVE_SPINUP_YEARS, tolerance=0, time_step=time_step)
        
        ## init df
        df = [None] * METOS_TRACER_DIM
        parameters_dim = len(parameters)
        h_factors_dim = len(h_factors)
        
        eps = np.spacing(1)
        eta = np.sqrt(eps) # square root of accuracy of F
        eta = 10**(-7) # square root of accuracy of F
        h = np.empty((parameters_dim, h_factors_dim))
        
        parameters_lower_bound = self.parameters_lower_bound
        parameters_upper_bound = self.parameters_upper_bound
        
#         process_pool = multiprocessing.pool.Pool(processes=parameters_dim)
#         process_pool.map(f, range(parameters_dim))
#         process_pool.close()
#         process_pool.join()
        ## for each parameter
        for parameter_index in range(parameters_dim):
            
            ## for each h
            for h_factor_index in range(h_factors_dim):
                
                ## prepare parameters
                parameters_der = np.copy(parameters)
                
                h[parameter_index, h_factor_index] = h_factors[h_factor_index] * max(abs(parameters[parameter_index]), 10**(-1)) * eta
                parameters_der[parameter_index] += h[parameter_index, h_factor_index]
                
                ## consider bounds
                violates_lower_bound = parameters_der[parameter_index] < parameters_lower_bound[parameter_index]
                violates_upper_bound = parameters_der[parameter_index] > parameters_upper_bound[parameter_index]
                
                if accuracy_order == 1:
                    if violates_lower_bound or violates_upper_bound:
                        h[parameter_index, h_factor_index] *= -1
                        parameters_der[parameter_index] = parameters[parameter_index] + h[parameter_index, h_factor_index]
                else:
                    if violates_lower_bound or violates_upper_bound:
                        if violates_lower_bound:
                            parameters_der[parameter_index] = parameters_lower_bound[parameter_index]
                        else:
                            parameters_der[parameter_index] = parameters_upper_bound[parameter_index]
                h[parameter_index, h_factor_index] = parameters_der[parameter_index] - parameters[parameter_index] # improvement of accuracy of h
                
                logger.debug('Calculating finite differences approximation for parameter index {} with h value {}.'.format(parameter_index, h[parameter_index, h_factor_index]))
                
                
#                 partial_derivative_dirname = util.pattern.replace_int_pattern(MODEL_PARTIAL_DERIVATIVE_DIRNAME, (parameter_index, h_factor_index))
                
                ## get previous run dir
                h_factor = int(np.sign(h[parameter_index, h_factor_index]))
                partial_derivative_dirname = MODEL_PARTIAL_DERIVATIVE_DIRNAME.format(parameter_index, h_factor)
                partial_derivative_dir = os.path.join(derivative_dir, partial_derivative_dirname)
                last_der_run_dir = self.get_last_run_dir(partial_derivative_dir)
                
                ## make new run if run not matching
                if not self.is_run_matching_options(last_der_run_dir, MODEL_DERIVATIVE_SPINUP_YEARS, 0, combination='or'):
                    util.io.remove_recursively(partial_derivative_dir, force=True, exclude_dir=True)
                    last_der_run_dir = self.make_run(partial_derivative_dir, parameters_der, MODEL_DERIVATIVE_SPINUP_YEARS, 0, time_step, tracer_input_path=spinup_run_dir)
                
                ## add trajectory to df
                trajectory = self._get_trajectory(load_trajectory_function, last_der_run_dir, parameters)
                for tracer_index in range(METOS_TRACER_DIM):
#                     with multiprocessing.Semaphore():
                    if df[tracer_index] is None:
                        df[tracer_index] = np.zeros((parameters_dim,) + trajectory[tracer_index].shape)
                    df[tracer_index][parameter_index] += (-1)**h_factor_index * trajectory[tracer_index]
            
            ## calculate df 
            for tracer_index in range(METOS_TRACER_DIM):
                if accuracy_order == 1:
                    df[tracer_index][parameter_index] -= f[tracer_index]
                    df[tracer_index][parameter_index] /= h[parameter_index]
                else:
                    df[tracer_index][parameter_index] /= np.sum(np.abs(h[parameter_index]))
        
        return df
    
    
    def df_all(self, parameters, time_dim_desired, years, tolerance=0, time_step=1, combination='or', accuracy_order=2):
        logger.debug('Calculating all df values for parameters {} and accuracy order {} with time dimension {}.'.format(parameters, accuracy_order, time_dim_desired))
        df = self._df(self._get_load_trajectory_function_for_all(time_dim_desired=time_dim_desired), parameters, years, tolerance=tolerance, combination=combination, time_step=time_step, accuracy_order=accuracy_order)
        
        assert len(df) == 2
        return df
    
    
    def df_points(self, parameters, points, years, tolerance=0, time_step=1, combination='or', accuracy_order=2):
        logger.debug('Calculating all df values for parameters {} and accuracy order {} at {} points.'.format(parameters, accuracy_order, tuple(map(len, points))))
        df = self._df(self._get_load_trajectory_function_for_points(points), parameters, years, tolerance=tolerance, combination=combination, time_step=time_step, accuracy_order=accuracy_order)
        
        assert len(df) == 2
        assert (not np.any(np.isnan(df[0]))) and (not np.any(np.isnan(df[1])))
        return df
    