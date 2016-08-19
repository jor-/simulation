import os
import numpy as np

import simulation.model.eval
import simulation.util.constants

import util.cache
import util.io.np
import util.parallel.with_multiprocessing

import util.logging
logger = util.logging.logger


class MemoryCache:

    def __init__(self, enabled=True):
        logger.debug('Initiating {} with enabled {}.'.format(self.__class__.__name__, enabled))
        self.memory_cache = util.cache.MemoryCache()
        self.last_parameters = None
        self.enabled = enabled


    def load_value(self, parameters, filename):
        if self.enabled and self.last_parameters is not None and np.allclose(parameters, self.last_parameters):
            return self.memory_cache.load_value(filename)
        else:
            raise util.cache.CacheMissError(filename)


    def save_value(self, parameters, filename, value):
        assert value is not None
        if self.enabled:
            if self.last_parameters is None or np.any(parameters != self.last_parameters):
                self.last_parameters = parameters
                self.memory_cache = util.cache.MemoryCache()
            self.memory_cache.save_value(filename, value)


    def get_value(self, parameters, filename, calculate_function):
        assert callable(calculate_function)

        try:
            value = self.load_value(parameters, filename)
        except util.cache.CacheMissError:
            value = calculate_function(parameters)
            self.save_value(parameters, filename, value)

        return value




class Cache:

    def __init__(self, model_options=None, cache_dirname=None, use_memory_cache=True):
        logger.debug('Initiating {} with cache dirname {}, model_options {} and use_memory_cache {}.'.format(self.__class__.__name__, cache_dirname, model_options, use_memory_cache))
        from simulation.model.constants import MODEL_SPINUP_MAX_YEARS

        ## prepare cache dirname
        if cache_dirname is None:
            cache_dirname = ''
        self.cache_dirname = cache_dirname

        ## prepare model
        self.model = simulation.model.eval.Model(model_options=model_options)

        ## prepare spinup options
        years = self.model.spinup_options['years']
        tolerance = self.model.spinup_options['tolerance']
        combination = self.model.spinup_options['combination']
        if combination == 'and':
            combination = True
        elif combination == 'or':
            combination = False
        elif not combination in (0, 1):
            raise ValueError('Combination "{}" unknown.'.format(combination))
        spinup_options = (years, tolerance, combination)
        self.desired_spinup_options = spinup_options
        
        ## prepare df options
        derivative_options = (self.model.derivative_options['years'], self.model.derivative_options['step_size'], self.model.derivative_options['accuracy_order'])
        self.derivative_options = derivative_options

        self.max_options = (MODEL_SPINUP_MAX_YEARS, 0, False) + derivative_options

        ## prepare memory cache
        self.memory_cache = MemoryCache(use_memory_cache)
        self.last_parameters = None


    def memory_cache_switch(self, enabled):
        self.memory_cache.enabled  = enabled
    
    
    def real_spinup_options(self, parameters):
        from simulation.model.constants import DATABASE_SPINUP_DIRNAME
        parameter_set_dir = self.parameter_set_dir(parameters)
        spinup_dir = os.path.join(parameter_set_dir, DATABASE_SPINUP_DIRNAME)
        last_run_dir = self.model.last_run_dir(spinup_dir)
        years = self.model.get_total_years(last_run_dir)
        tolerance = self.model.get_real_tolerance(last_run_dir)
        spinup_options = (years, tolerance, True)
        return spinup_options
    
    
    def desired_options(self, parameters):
        if self.desired_spinup_options is not None:
            desired_options = self.desired_spinup_options
        else:
            desired_options = self.real_spinup_options(parameters)
        desired_options = desired_options + self.derivative_options
        
        assert len(desired_options) == 6
        return desired_options



    ## access to cache

    def parameter_set_dir(self, parameters):
        VALUE_NAME = 'parameter_set_dir'
        try:
            parameter_set_dir = self.memory_cache.load_value(parameters, VALUE_NAME)
        except util.cache.CacheMissError:
            parameter_set_dir = self.model.parameter_set_dir(parameters, create=False)
            if parameter_set_dir is not None:
                self.memory_cache.save_value(parameters, VALUE_NAME, parameter_set_dir)

        return parameter_set_dir


    def get_file(self, parameters, filename):
        assert filename is not None
        parameter_set_dir = self.parameter_set_dir(parameters)

        if parameter_set_dir is not None:
            cache_dir = os.path.join(parameter_set_dir, self.cache_dirname)
            file = os.path.join(cache_dir, filename)
        else:
            file = None

        return file
    
    
    def options_filename(self, filename):
        filename_root, filename_ext = os.path.splitext(filename)
        option_filename = filename_root + simulation.util.constants.OPTION_FILE_SUFFIX + filename_ext
        return option_filename
        
    

    def matches_options(self, parameters, filename):
        options_filename = self.options_filename(filename)
        loaded_options = self.load_value(parameters, options_filename)
        desired_options = self.desired_options(parameters)

        def is_matching(loaded_options, desired_options):
            assert loaded_options is None or len(loaded_options) in [3, 6]
            assert len(desired_options) == 6
            
            ## if previous value is available, check its options.
            if loaded_options is not None:
                
                ## check spinup options
                matches_year = desired_options[0] <= loaded_options[0] or np.isclose(desired_options[0], loaded_options[0])
                matches_tolerance = desired_options[1] >= loaded_options[1] or np.isclose(desired_options[1], loaded_options[1])
                if loaded_options[2]:
                    if desired_options[2]:
                        matches = matches_year and matches_tolerance
                    else:
                        matches = matches_year or matches_tolerance
                else:
                    if desired_options[2]:
                        matches = False
                    else:
                        matches = matches_year and matches_tolerance

                ## check derivative options
                if len(loaded_options) == 6:
                    matches_year = desired_options[3] <= loaded_options[3] or np.isclose(desired_options[3], loaded_options[3])
                    matches_step_size = np.isclose(desired_options[4], loaded_options[4])
                    matches_accuracy_order = desired_options[5] <= loaded_options[5] or np.isclose(desired_options[5], loaded_options[5])
                    matches = matches and matches_year and matches_step_size and matches_accuracy_order
            
            ## if no previous value with options is available, return False.
            else:
                matches = False
            
            return matches

        matches_options = is_matching(loaded_options, desired_options)
        logger.debug('Loaded options {} matches desired options {}: {}.'.format(loaded_options, desired_options, matches_options))

        if not matches_options:
            matches_options = is_matching(loaded_options, self.max_options)
            logger.debug('Loaded options {} matches max options {}: {}.'.format(loaded_options, self.max_options, matches_options))

        return matches_options


    ## value
    
    def load_value(self, parameters, filename, use_memmap=False, as_shared_array=False):
        file = self.get_file(parameters, filename)
        if file is not None and os.path.exists(file):
            if use_memmap or as_shared_array:
                mem_map_mode = 'r'
            else:
                mem_map_mode = None
            logger.debug('Loading value from {} with mem_map_mode {} and as_shared_array {}.'.format(file, mem_map_mode, as_shared_array))
            value = np.load(file, mmap_mode=mem_map_mode)
            if as_shared_array:
                value = util.parallel.with_multiprocessing.shared_array(value)
        else:
            value = None
        return value


    def _save_value_without_options(self, parameters, filename, value, save_also_txt=True):
        ## check inpput
        if value is None:
            raise ValueError('Value for {} with parameters {} is None!'.format(filename, parameters))
        if filename is None:
            raise ValueError('Filename for parameters {} is None!'.format(parameters))
        
        ## save value
        file = self.get_file(parameters, filename)
        logger.debug('Saving value to {} file with save_also_txt {}.'.format(file, save_also_txt))
        assert file is not None
        os.makedirs(os.path.dirname(file), exist_ok=True)
        assert file is not None
        if save_also_txt:
            util.io.np.save_npy_and_txt(file, value, make_read_only=True, overwrite=True)
        else:
            util.io.np.save(file, value, make_read_only=True, overwrite=True)


    def save_value(self, parameters, filename, value, derivative_used=True, save_also_txt=True):
        ## save value
        self._save_value_without_options(parameters, filename, value, save_also_txt=save_also_txt)
        
        ## save option
        options = self.real_spinup_options(parameters)
        assert len(options) == 3
        if derivative_used:
            options = options + self.derivative_options
            assert len(options) == 6
        
        option_filename = self.options_filename(filename)
        self._save_value_without_options(parameters, option_filename, options, save_also_txt=True)

    
    def get_value(self, parameters, filename, calculate_function, derivative_used=True, save_also_txt=True, use_memmap=False, as_shared_array=False):

        assert callable(calculate_function)

        ## try to load from memory cache
        try:
            value = self.memory_cache.load_value(parameters, filename)

        ## if not found try to load from file or calculate
        except util.cache.CacheMissError:
            is_matchig = self.matches_options(parameters, filename)

            ## if not matching calculate and save value
            if not is_matchig:
                ## calculating and saving value
                logger.debug('Calculating value with {} and saving with filename {} with derivative_used {}.'.format(calculate_function, filename, derivative_used))
                value = calculate_function(parameters)
                self.save_value(parameters, filename, value, derivative_used=derivative_used, save_also_txt=save_also_txt)

            ## load value if matching or memmap used
            if is_matchig or use_memmap or as_shared_array:
                value = self.load_value(parameters, filename, use_memmap=use_memmap, as_shared_array=as_shared_array)

            ## update memory cache
            self.memory_cache.save_value(parameters, filename, value)

        return value


    def has_value(self, parameters, filename):
        return self.matches_options(parameters, filename)



