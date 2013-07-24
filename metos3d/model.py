import os
import re
import warnings
import tempfile
import stat
import time
import numpy as np

import util.pattern
import util.io
from util.debug import Debug

import ndop.metos3d.data
from ndop.metos3d.job import Job

class Model(Debug):
    
    def __init__(self, debug_level=0, required_debug_level=1):
        from ndop.metos3d.constants import  MODEL_PARAMETER_LOWER_BOUND, MODEL_PARAMETER_UPPER_BOUND
        
        Debug.__init__(self, debug_level, required_debug_level-1, 'ndop.metos3d.model: ')
        
        self.parameters_lower_bound = MODEL_PARAMETER_LOWER_BOUND
        self.parameters_upper_bound = MODEL_PARAMETER_UPPER_BOUND
    
    
    
    @property
    def land_sea_mask(self):
        """The land-sea-mask from metos3d as numpy array."""
        
        self.print_debug_inc('Getting land-sea-mask.')
        
        ## return land-sea-mask if loaded
        try:
            land_sea_mask = self.__land_sea_mask
            self.print_debug('Returning cached land-sea-mask.')
        ## otherwise load land-sea-mask and return it
        except AttributeError:
            land_sea_mask = ndop.metos3d.data.load_land_sea_mask(self.debug_level, self.required_debug_level)
            self.__land_sea_mask = land_sea_mask
        
        self.print_debug_dec('Got land-sea-mask.')
        
        return land_sea_mask
    
    
    
    def check_if_parameters_in_bounds(self, parameters):
        if any(parameters < self.parameters_lower_bound):
            indices = np.where(parameters < self.parameters_lower_bound)
            string = ''
            for i in indices:
                string += str(parameters[i]) + '<' + str(parameters_lower_bound[i])
                if i < len(indices - 1):
                    string += ', '
            raise ValueError('Some parameters are lower than the lower bound: ' + string)
        
        if any(parameters > self.parameters_upper_bound):
            indices = np.where(parameters > self.parameters_upper_bound)
            string = ''
            for i in indices:
                string += str(parameters[i]) + '>' + str(parameters_upper_bound[i])
                if i < len(indices - 1):
                    string += ', '
            raise ValueError('Some parameters are upper than the upper bound: ' + string)
    
    
    
    def run_job(self, model_parameters, output_path=os.getcwd(), years=1, tolerance=0, time_step_size=1, write_trajectory=False, tracer_input_path=None, pause_time_seconds=10):
        
        self.print_debug_inc(('Running job with years = ', years, ' tolerance = ', tolerance, ' time_step_size = ', time_step_size, ' tracer_input_path = ', tracer_input_path))
        
        ## check parameters
        self.check_if_parameters_in_bounds(model_parameters)
        
        ## execute job
        job = Job(debug_level=self.debug_level, required_debug_level=self.required_debug_level+1)
        job.initialise_with_best_configuration(model_parameters, output_path=output_path, years=years, tolerance=tolerance, time_step_size=time_step_size, write_trajectory=write_trajectory, tracer_input_path=tracer_input_path, pause_time_seconds=pause_time_seconds)
        time.sleep(2)
        job.start()
        job.wait_until_finished()
        
        ## change access mode of files
        for (dirpath, dirnames, filenames) in os.walk(output_path, topdown=False):
            for filename in filenames:
                file = os.path.join(dirpath, filename)
                os.chmod(file, stat.S_IRUSR)
            for dirname in dirnames:
                dir = os.path.join(dirpath, dirname)
                os.chmod(dir, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        
        self.print_debug_dec('Job finished.')
    
    
    def search_or_make_time_step_dir(self, search_path, time_step):
        from ndop.metos3d.constants import MODEL_TIME_STEP_DIRNAME
        
        self.print_debug_inc(('Searching for directory for time step size "', time_step, '" in "', search_path, '".'))
        
        time_step_dirname = util.pattern.replace_int_pattern(MODEL_TIME_STEP_DIRNAME, time_step)
        time_step_dir = os.path.join(search_path, time_step_dirname)
        
        try:
            os.makedirs(time_step_dir)
            self.print_debug_dec(('Directory "', time_step_dir, '" created.'))
        except OSError:
            self.print_debug_dec(('Directory "', time_step_dir, '" found.'))
        
        return time_step_dir
    
    
    
    def get_parameters_diff(self, parameters, parameter_set_dir):
        from ndop.metos3d.constants import MODEL_PARAMETERS_FILENAME
        
        parameters_file = os.path.join(parameter_set_dir, MODEL_PARAMETERS_FILENAME)
        current_parameters = np.loadtxt(parameters_file)
        
        parameter_diff = np.linalg.norm(parameters - current_parameters)
        
        return parameter_diff
    
    
    
#     def get_parameter_set_dir(self, path):
#         from ndop.metos3d.constants import MODEL_PARAMETERS_SET_DIRNAME
#         
#         parameter_set_dir = None
#         while parameter_set_dir is None and len(path) > 0:
#             (head, tail) = os.path.split(path)
#             if util.pattern.is_matching(tail, MODEL_PARAMETERS_SET_DIRNAME):
#                 parameter_set_dir = path
#             else:
#                 path = head
#         
#         return parameter_set_dir
    
    
    def is_matching_parameter_set(self, parameters, parameter_set_dir):
        from ndop.metos3d.constants import MODEL_PARAMETERS_MAX_DIFF
        
#         parameter_set_dir = self.get_parameter_set_dir(path)
        parameters_diff = self.get_parameters_diff(parameters, parameter_set_dir)
        is_matching = parameters_diff <= MODEL_PARAMETERS_MAX_DIFF
        
        return is_matching
    
#     def is_same_parameter_set(self, path1, path2):
#         parameter_set_dir_1 = self.get_parameter_set_dir(path1)
#         parameter_set_dir_2 = self.get_parameter_set_dir(path2)
#         
#         is_matching = parameter_set_dir_1 == parameter_set_dir_2
#         
#         return is_matching
    
    
#     def search_nearest_parameter_set_dir(self, search_path, parameters):
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
#             while parameters_diff > 0 and parameter_set_number < len(parameter_set_dirs):
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
#         if parameter_set_dir is not None:
#             self.print_debug_dec(('Directory closest to parameters at "', parameter_set_dir, '" found.'))
#         else:
#             self.print_debug_dec('No directory closest to parameters found.')
#         
#         return parameter_set_dir
    
    
    
    def make_new_parameter_set_dir(self, path, parameters):
        from ndop.metos3d.constants import MODEL_PARAMETERS_SET_DIRNAME, MODEL_PARAMETERS_FILENAME, MODEL_PARAMETERS_FORMAT_STRING
        
        self.print_debug_inc(('Creating new parameter set directory in "', path, '".'))
        
#         parameter_set_dirs = util.io.get_dirs(path)
#         next_parameter_set_number = len(parameter_set_dirs)
#         
#         parameter_set_dirname = util.pattern.replace_int_pattern(MODEL_PARAMETERS_SET_DIRNAME, next_parameter_set_number)
#         parameter_set_dir = os.path.join(path, parameter_set_dirname)
#         os.makedirs(parameter_set_dir)

        parameter_set_number = 0
        parameter_set_dir = None
        while parameter_set_dir is None:
            parameter_set_dirname = util.pattern.replace_int_pattern(MODEL_PARAMETERS_SET_DIRNAME, parameter_set_number)
            parameter_set_dir_candidate = os.path.join(path, parameter_set_dirname)
            if os.path.isdir(parameter_set_dir_candidate):
                parameter_set_number +=  1
            else:
                os.makedirs(parameter_set_dir_candidate)
                parameter_set_dir = parameter_set_dir_candidate
        
        parameters_file = os.path.join(parameter_set_dir, MODEL_PARAMETERS_FILENAME)
        np.savetxt(parameters_file, parameters, fmt=MODEL_PARAMETERS_FORMAT_STRING)
        os.chmod(parameters_file, stat.S_IRUSR)
        self.print_debug_dec(('New parameter set directory "', parameter_set_dir, '" created.'))
        
        return parameter_set_dir
        
        
    def search_or_make_parameter_set_dir(self, search_path, parameters):
        self.print_debug_inc(('Searching for directory for parameters "', parameters, '" in "', search_path, '".'))
        
        ## check input
        self.check_if_parameters_in_bounds(parameters)
        
        ## search for directories with matching parameters
        parameter_set_dir = None
        parameter_set_number = 0
        
        if os.path.exists(search_path):
# #             closest_parameter_set_dir = self.search_nearest_parameter_set_dir(search_path, parameters)
# #             if self.is_matching_parameter_set(parameters, closest_parameter_set_dir):
# #                 parameter_set_dir = closest_parameter_set_dir
# # #             closest_parameters = np.loadtxt(current_parameters_file)
# # #             if all(closest_parameters == parameters):
# # #                 parameter_set_dir = closest_parameter_set_dir
            parameter_set_dirs = util.io.get_dirs(search_path)
            while parameter_set_dir is None and parameter_set_number < len(parameter_set_dirs):
                current_parameter_set_dir = parameter_set_dirs[parameter_set_number]
#                 current_parameters_file = os.path.join(current_parameter_set_dir, MODEL_PARAMETERS_FILENAME)
                try:
#                     current_parameters = np.loadtxt(current_parameters_file)
                    if self.is_matching_parameter_set(parameters, current_parameter_set_dir):
                    #if all(current_parameters == parameters):
                        parameter_set_dir = current_parameter_set_dir
                except (OSError, IOError):
                    warnings.warn('Could not read the parameters file "' + current_parameters_file + '"!')
                
                if parameter_set_dir is None:
                        parameter_set_number += 1
        else:
            os.makedirs(search_path)
        
        
        ## make new model_output if the parameters are not matching
        if parameter_set_dir is not None:
            self.print_debug_dec(('Matching directory for parameters at "', current_parameter_set_dir, '" found.'))
        else:
            parameter_set_dir = self.make_new_parameter_set_dir(search_path, parameters)
#             self.print_debug(('No matching directory for parameters found.'))
#             parameter_set_dirname = util.pattern.replace_int_pattern(MODEL_PARAMETERS_SET_DIRNAME, parameter_set_number)
#             parameter_set_dir = os.path.join(search_path, parameter_set_dirname)
#             os.makedirs(parameter_set_dir)
#             
#             parameters_file = os.path.join(parameter_set_dir, MODEL_PARAMETERS_FILENAME)
#             np.savetxt(parameters_file, parameters, fmt=MODEL_PARAMETERS_FORMAT_STRING)
#             os.chmod(parameters_file, stat.S_IRUSR)
#             self.print_debug(('Directory "', parameter_set_dir, '" for parameters created.'))
            
            self.print_debug_dec(('Directory for parameters "', parameters, '" created in "', parameter_set_dir, '".'))
        
        return parameter_set_dir
    
    
    def get_run_dirs(self, search_path):
        from ndop.metos3d.constants import MODEL_RUN_DIRNAME
        
        run_dir_condition = lambda file: os.path.isdir(file) and util.pattern.is_matching(os.path.basename(file), MODEL_RUN_DIRNAME)
        try:
            run_dirs = util.io.filter_files(search_path, run_dir_condition)
        except (OSError, IOError) as exception:
            warnings.warn('It could not been searched in the search path "' + search_path + '": ' + str(exception))
            run_dirs = []
        
        return run_dirs
    
    
    
    def search_last_run_dir(self, search_path):
        from ndop.metos3d.constants import MODEL_RUN_DIRNAME
        
        self.print_debug_inc(('Searching for last run in "', search_path, '".'))
        
        run_dirs = self.get_run_dirs(search_path)
        
        last_run_dir = None
        last_run_index =  len(run_dirs) - 1
        
        while last_run_dir is None and last_run_index >= 0:
            last_run_dir = run_dirs[last_run_index]
            self.print_debug(('Searching in "', last_run_dir, '".'))
            
            # load last job finished
            if last_run_dir is not None:
                job = Job(debug_level=self.debug_level, required_debug_level=self.required_debug_level+1)
                try:
                    job.load(last_run_dir)
                except (OSError, IOError) as exception:
                    warnings.warn('Could not read the job options file from "' + last_run_dir + '": ' + str(exception))
                    last_run_dir = None
            
            # check if job is finished
            if last_run_dir is not None:
                try:
                    job.wait_until_finished()
                except (OSError, IOError) as exception:
                    warnings.warn('Could not check if job ' + job.id + ' is finished: ' + str(exception))
                    last_run_dir = None
            
            last_run_index -= 1
        
        self.print_debug_dec('Run found.')
        
        return last_run_dir
    
    
    
#     def get_run_options(self, run_path):
#         from ndop.metos3d.constants import MODEL_RUN_OPTIONS_FILENAME
#         
#         options_file = os.path.join(run_path, MODEL_RUN_OPTIONS_FILENAME)
#         options = np.loadtxt(options_file)
#         
#         return options
    
    
    
    def is_run_matching_options(self, run_dir, years, tolerance, time_step_size):
        if run_dir is not None:
#             (run_years, run_tolerance, run_time_step_size) = self.get_run_options(run_path
#             is_matching = years <= run_years and tolerance >= run_tolerance and time_step_size >= run_time_step_size

            run_years = self.get_total_years(run_dir)
            run_tolerance = self.get_real_tolerance(run_dir)
            run_time_step_size = self.get_time_step_size(run_dir)
            
            
            is_matching = time_step_size >= run_time_step_size and (years <= run_years or tolerance >= run_tolerance)
        else:
            is_matching = False
        
        return is_matching
    
    
    
    def previous_run_dir(self, run_dir):
        from ndop.metos3d.constants import MODEL_RUN_DIRNAME
        
        (spinup_dir, run_dirname) = os.path.split(run_dir)
        run_index = int(re.findall('\d+', run_dirname)[0])
        if run_index > 0:
            previous_run_dirname = util.pattern.replace_int_pattern(MODEL_RUN_DIRNAME, run_index - 1)
            previous_run_dir = os.path.join(spinup_dir, previous_run_dirname)
        else:
            previous_run_dir = None
        
        return previous_run_dir
    
    
    
    def get_total_years(self, run_dir):
        total_years = 0
        
        while run_dir is not None:
            run_job = Job(debug_level=self.debug_level, required_debug_level=self.required_debug_level+1)
            run_job.load(run_dir)
            run_years = run_job.last_year
            
            total_years += run_years
            
            run_dir = self.previous_run_dir(run_dir)
        
#         for run_dir in self.get_run_dirs(search_path):
#             run_job = Job(debug_level=self.debug_level, required_debug_level=self.required_debug_level+1)
#             run_job.load(run_dir)
#             run_years = run_job.last_year
#             
#             total_years += run_years
        
        return total_years
    
    
    
    def get_time_step_size(self, run_dir):
        from ndop.metos3d.constants import MODEL_RUN_OPTIONS_FILENAME
        
        options_file = os.path.join(run_dir, MODEL_RUN_OPTIONS_FILENAME)
        options = np.loadtxt(options_file)
        
        (years, tolerance, time_step_size) = options
        
        return time_step_size
    
    
    
    def get_real_tolerance(self, run_dir):
        job = Job(debug_level=self.debug_level, required_debug_level=self.required_debug_level+1)
        job.load(run_dir)
        tolerance = job.last_tolerance     
        
        return tolerance
    
    
    
    def get_tracer_input_dir(self, run_dir):
        job = Job(debug_level=self.debug_level, required_debug_level=self.required_debug_level+1)
        job.load(run_dir)
        tracer_input_dir = job.get_tracer_input_path()
        
        return tracer_input_dir
    
    
#     def extract_parameter_set_dir(self, path):
#         from ndop.metos3d.constants import MODEL_PARAMETERS_SET_DIRNAME
#         
#         path_split = path.split(os.path.sep)
#         match_list = util.pattern.get_all_matching_strings(path_split, MODEL_PARAMETERS_SET_DIRNAME)
#         
#         if len(match_list) >= 1:
#             return match_list[0]
#         else:
#             return None

#     def get_parameters(self, path):
#         parameters = None
#         
#         parameters_file =
#         os.path.dirname(path)
#         
#         
#         from ndop.metos3d.constants import MODEL_PARAMETERS_FILENAME, MODEL_PARAMETERS_SET_DIRNAME, MODEL_PARAMETERS_FORMAT_STRING
#         
#         self.print_debug_inc(('Searching for directory for parameters "', parameters, '" in "', search_path, '".'))
#         
#         parameter_set_dirname = util.pattern.replace_int_pattern(MODEL_PARAMETERS_SET_DIRNAME, parameter_set_number)
#         parameter_set_dir = os.path.join(search_path, parameter_set_dirname)
#         os.makedirs(parameter_set_dir)
#         
#         parameters_file = os.path.join(parameter_set_dir, MODEL_PARAMETERS_FILENAME)
#         np.savetxt(parameters_file, parameters, fmt=MODEL_PARAMETERS_FORMAT_STRING)
    
    
    def make_run(self, output_path, parameters, years, tolerance, time_step_size, tracer_input_path=None):
        from ndop.metos3d.constants import MODEL_RUN_DIRNAME, MODEL_RUN_OPTIONS_FILENAME
        
        self.print_debug_inc('Creating new run directory ...')
        
        ## get next run index
        util.io.makedirs_if_not_exists(output_path)
        run_dirs = self.get_run_dirs(output_path)
        next_run_index = len(run_dirs)
        
        ## create run dir
        run_dirname = util.pattern.replace_int_pattern(MODEL_RUN_DIRNAME, next_run_index)
        run_dir = os.path.join(output_path, run_dirname)
        self.print_debug(('... at ', run_dir, ' ...'))
        os.makedirs(run_dir)
        
        ## create run options file
        run_options = np.array((years, tolerance, time_step_size))
        run_options_file = os.path.join(run_dir, MODEL_RUN_OPTIONS_FILENAME)
        np.savetxt(run_options_file, run_options)
        os.chmod(run_options_file, stat.S_IRUSR)
        
        ## create run
        if tracer_input_path is not None and output_path == os.path.dirname(tracer_input_path):
        # and self.is_same_parameter_set(output_path, tracer_input_path):
            last_years = self.get_total_years(tracer_input_path)
        else:
            last_years = 0
        self.print_debug(('... with last_years = ', last_years, ' and tracer_input_path = ', tracer_input_path))
            
        self.run_job(parameters, output_path=run_dir, tracer_input_path=tracer_input_path, years=years-last_years, tolerance=tolerance, time_step_size=time_step_size, pause_time_seconds=150)
        
        self.print_debug_dec(('Run directory ', run_dir, ' created.'))
        
        return run_dir
    
    
    
    def search_closest_spinup_run(self, path, parameters):
        from ndop.metos3d.constants import MODEL_SPINUP_DIRNAME
        self.print_debug_inc(('Searching for spinup run as close as possible to parameters "', parameters, '" in "', path, '".'))
        
        ## search for parameter set directories with matching parameters
        closest_run_dir = None
        parameter_set_number = 0
        parameters_diff = float('inf')
        
        parameter_set_dirs = util.io.get_dirs(path)
        while parameters_diff > 0 and parameter_set_number < len(parameter_set_dirs):
            current_parameter_set_dir = parameter_set_dirs[parameter_set_number]
            try:
                current_parameters_diff = self.get_parameters_diff(parameters, current_parameter_set_dir)
            except (OSError, IOError):
                warnings.warn('Could not read the parameters file "' + current_parameters_file + '"!')
                current_parameters_diff = float('inf')
            
            current_spinup_dir = os.path.join(current_parameter_set_dir, MODEL_SPINUP_DIRNAME)
            last_run_dir = self.search_last_run_dir(current_spinup_dir)
            if last_run_dir is not None and current_parameters_diff < parameters_diff:
                closest_run_dir = last_run_dir
                parameters_diff = current_parameters_diff
            
            parameter_set_number += 1
        
        if closest_run_dir is not None:
            self.print_debug_dec(('Spinup run as close as possible found at "', closest_run_dir, '".'))
        else:
            self.print_debug_dec('No spinup run found.')
        
        return closest_run_dir
    
    
    
    def search_or_make_spinup(self, parameter_set_dir, parameters, years, tolerance, time_step_size):
        from ndop.metos3d.constants import MODEL_SPINUP_DIRNAME
        
        spinup_dir = os.path.join(parameter_set_dir, MODEL_SPINUP_DIRNAME)
        
        self.print_debug_inc(('Searching for spinup with "', years, '" years, "', tolerance, '" tolerance and "', time_step_size, '" time step size in "', spinup_dir, '".'))
        
        last_run_dir = self.search_last_run_dir(spinup_dir)
        
        ## matching spinup found
        if self.is_run_matching_options(last_run_dir, years, tolerance, time_step_size):
            run_dir = last_run_dir
            self.print_debug(('Matching spinup at ', last_run_dir, ' found.'))
            
        ## create new spinup
        else:
            self.print_debug(('No matching spinup found.'))
            output_dir = os.path.dirname(parameter_set_dir)
            last_run_dir = self.search_closest_spinup_run(output_dir, parameters)
        
            run_dir = self.make_run(spinup_dir, parameters, years, tolerance, time_step_size, tracer_input_path=last_run_dir)
            
            self.print_debug(('Spinup directory ', run_dir, ' created.'))
            
            self.remove_f(parameter_set_dir)
        
        self.print_debug_dec('Spinup found.')
        
        return run_dir
    
    
    
    
    def get_trajectory(self, run_dir, parameters, t_dim, time_step_size):
#         (run_years, run_tolerance, run_time_step_size) = self.get_run_options(run_dir)
        run_time_step_size = self.get_time_step_size(run_dir)
        
        with tempfile.TemporaryDirectory(dir=run_dir, prefix='trajectory_tmp_') as tmp_path:
            self.run_job(parameters, output_path=tmp_path, tracer_input_path=run_dir, write_trajectory=True, years=1, tolerance=0, time_step_size=run_time_step_size, pause_time_seconds=10)
            
            trajectory_path = os.path.join(tmp_path, 'trajectory')
            
            trajectory = ndop.metos3d.data.load_trajectories(trajectory_path, t_dim, time_step_size, land_sea_mask=self.land_sea_mask, debug_level=self.debug_level, required_debug_level=self.required_debug_level + 1)
        
        return trajectory
    
    
    
    def search_or_make_f(self, search_path, spinup_dir, parameters, t_dim, time_step_size):
        from ndop.metos3d.constants import MODEL_F_FILENAME, MODEL_TIME_STEP_SIZE_MAX
        
        self.print_debug_inc(('Searching for F file with time dimension "', t_dim, '" in "', search_path, '".'))
        
        f = None
        
        ## searching for appropriate F file
        if os.path.exists(search_path):
            f_filename = util.pattern.replace_int_pattern(MODEL_F_FILENAME, t_dim)
            f_file = os.path.join(search_path, f_filename)
            try:
                f = np.load(f_file)
                self.print_debug(('Matching F file found: "', f_file, '".'))
            except (OSError, IOError): #FileNotFoundError:
                pass
        else:
            os.makedirs(search_path)
        
        
        ## make new F file if appropriate F file is missing
        if f is None:
            self.print_debug('No matching F file found.')
            f = self.get_trajectory(spinup_dir, parameters, t_dim, time_step_size)
            np.save(f_file, f)
            self.print_debug(('F file ', f_file, ' created.'))
        
        self.print_debug_dec(('Got F file "', f_file, '".'))
        
        return f
    
    
    
    def remove_f(self, search_path):
        from ndop.metos3d.constants import MODEL_F_FILENAME
        
        self.print_debug_inc_dec(('Removing F files in "', search_path, '".'))
        
        f_file_condition = lambda file: os.path.isfile(file) and util.pattern.is_matching(os.path.basename(file), MODEL_F_FILENAME)
        f_files = util.io.filter_files(search_path, f_file_condition)
        for f_file in f_files:
            os.remove(f_file)
    
    
    
    
    def f(self, parameters, t_dim=12, years=7000, tolerance=0, time_step_size=1):
        from ndop.metos3d.constants import MODEL_OUTPUTS_DIR
        
        self.print_debug_inc(('Searching for f value for parameters "', parameters, '" in "', MODEL_OUTPUTS_DIR, '".'))
        
        time_step_dir = self.search_or_make_time_step_dir(MODEL_OUTPUTS_DIR, time_step_size)
        parameter_set_dir = self.search_or_make_parameter_set_dir(time_step_dir, parameters)
        spinup_run_dir = self.search_or_make_spinup(parameter_set_dir, parameters, years, tolerance, time_step_size)
        f = self.search_or_make_f(parameter_set_dir, spinup_run_dir, parameters, t_dim, time_step_size)
        
        self.print_debug_dec('F value found.')
        
        return f
    
    
    
    def remove_df(self, search_path):
        from ndop.metos3d.constants import MODEL_DF_FILENAME
        
        self.print_debug_inc_dec(('Removing DF files in "', search_path, '".'))
        
        file_condition = lambda file: os.path.isfile(file) and util.pattern.is_matching(os.path.basename(file), MODEL_DF_FILENAME)
        files = util.io.filter_files(search_path, file_condition)
        for file in files:
            os.remove(file)
    
    
    
    
    def df(self, parameters, t_dim=12, years=7000, tolerance=0, time_step_size=1, accuracy_order=1):
        from ndop.metos3d.constants import MODEL_OUTPUTS_DIR, MODEL_DERIVATIVE_DIRNAME, MODEL_SPINUP_DIRNAME, MODEL_PARTIAL_DERIVATIVE_DIRNAME, MODEL_DERIVATIVE_SPINUP_YEARS, MODEL_DF_FILENAME
        
        self.print_debug_inc(('Searching for derivative for parameters "', parameters, '" in "', MODEL_OUTPUTS_DIR, '".'))
        
        ## chose h factors
        if accuracy_order == 1:
            h_factors =(1,)
        elif accuracy_order == 2:
            h_factors =(1, -1)
        else:
            raise ValueError('Accuracy order ' + str(accuracy_order) + ' not supported.')
        
        ## search directories
        time_step_dir = self.search_or_make_time_step_dir(MODEL_OUTPUTS_DIR, time_step_size)
        parameter_set_dir = self.search_or_make_parameter_set_dir(time_step_dir, parameters)
        derivative_dir = os.path.join(parameter_set_dir, MODEL_DERIVATIVE_DIRNAME)
        
        ## check if spinup runs for derivatives are matching the options
        parameters_len = len(parameters)
        h_factors_len = len(h_factors)
        
        self.print_debug(('Checking if derivative spinup runs are matching "', years, '" years, "', tolerance, '" tolerance and "', time_step_size, '" time step size in "', derivative_dir, '".'))
        i = 0
        is_matching = True
        while is_matching and i < parameters_len:
            j = 0
            while is_matching and j < h_factors_len:
                ## check derivative spinup
                partial_derivative_dirname = util.pattern.replace_int_pattern(MODEL_PARTIAL_DERIVATIVE_DIRNAME, (i, j))
                partial_derivative_dir = os.path.join(derivative_dir, partial_derivative_dirname)
                last_der_run_dir = self.search_last_run_dir(partial_derivative_dir)
#                 is_matching = self.is_run_matching_options(last_der_run_dir, years, tolerance, time_step_size)
                is_matching = self.is_run_matching_options(last_der_run_dir, MODEL_DERIVATIVE_SPINUP_YEARS, 0, time_step_size)
                
                ## check normal input spinup
                if is_matching:
                    tracer_input_dir = self.get_tracer_input_dir(last_der_run_dir)
                    is_matching = is_matching and self.is_run_matching_options(tracer_input_dir, years - MODEL_DERIVATIVE_SPINUP_YEARS, tolerance, time_step_size)
                j += 1
            i += 1
        
        ## load DF File if available
        df = None
        if is_matching:
            self.print_debug('Existing derivative spinup runs are matching the options.')
            accuracy_order_i = 2
            while df is None and accuracy_order_i >= accuracy_order:
                df_filename = util.pattern.replace_int_pattern(MODEL_DF_FILENAME, (t_dim, accuracy_order_i))
                df_file = os.path.join(parameter_set_dir, df_filename)
                try:
                    df = np.load(df_file)
                    self.print_debug(('DF file "', df_file, '" loaded.'))
                except (OSError, IOError):
                    accuracy_order_i -= 1
        else:
            self.print_debug('Existing derivative spinup runs are not matching the options.')
        
        
        ## calculate DF if not available
        if df is None:
            self.print_debug('Calculating derivative.')
            self.remove_df(parameter_set_dir)
            
            spinup_run_dir = self.search_or_make_spinup(parameter_set_dir, parameters, years - MODEL_DERIVATIVE_SPINUP_YEARS, tolerance, time_step_size)
            spinup_run_years = self.get_total_years(spinup_run_dir)
            previous_spinup_run_dir = self.previous_run_dir(spinup_run_dir)
            previous_spinup_run_years = self.get_total_years(previous_spinup_run_dir)
            
            if previous_spinup_run_years == spinup_run_years - MODEL_DERIVATIVE_SPINUP_YEARS:
                spinup_run_dir = previous_spinup_run_dir
                spinup_run_years = previous_spinup_run_years
            
            if accuracy_order == 1:
                f = self.f(parameters, t_dim, spinup_run_years + MODEL_DERIVATIVE_SPINUP_YEARS, 0, time_step_size)
            else:
                f = self.f(parameters, t_dim, spinup_run_years, tolerance, time_step_size)
            df = np.zeros(f.shape + (parameters_len,))
            
            eps = np.spacing(1)
            eps_sqrt = np.sqrt(eps)
            h = np.empty(parameters_len)
            
            for i in range(parameters_len):
                h[i] = max(abs(parameters[i]), eps_sqrt * 2**8) * eps_sqrt
                
                for j in range(h_factors_len):
                    h_factor = h_factors[j]
                    partial_derivative_dirname = util.pattern.replace_int_pattern(MODEL_PARTIAL_DERIVATIVE_DIRNAME, (i, j))
                    partial_derivative_dir = os.path.join(derivative_dir, partial_derivative_dirname)
                    last_der_run_dir = self.search_last_run_dir(partial_derivative_dir)
                    
                    if not self.is_run_matching_options(last_der_run_dir, MODEL_DERIVATIVE_SPINUP_YEARS, 0, time_step_size):
                        delta_p_i = h_factor * h[i]
                        
                        parameters_der = np.copy(parameters)
                        parameters_der[i] += delta_p_i
                        parameters_der[i] = min(parameters_der[i], self.parameters_upper_bound[i])
                        parameters_der[i] = max(parameters_der[i], self.parameters_lower_bound[i])
                        
                        last_der_run_dir = self.make_run(partial_derivative_dir, parameters_der, MODEL_DERIVATIVE_SPINUP_YEARS, 0, time_step_size, tracer_input_path=spinup_run_dir)
                        
                    df[..., i] += h_factor * self.get_trajectory(last_der_run_dir, parameters, t_dim, time_step_size)
                
                if accuracy_order == 1:
                    df[..., i] -= f
                    df[..., i] /= h[i]
                else:
                    df[..., i] /= 2 * h[i]
            
            self.print_debug('Derivative calculated.')
            
            df_filename = util.pattern.replace_int_pattern(MODEL_DF_FILENAME, (t_dim, accuracy_order))
            df_file = os.path.join(parameter_set_dir, df_filename)
            self.print_debug(('Saving derivative at file"', df_file, '".'))
            np.save(df_file, df)
            
        self.print_debug_dec('Derivative found.')
        
        return df
        