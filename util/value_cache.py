import os
import numpy as np

from ndop.model.eval import Model

import util.io
import util.logging
logger = util.logging.get_logger()

import util.parallel.with_multiprocessing



class Cache:
    
    def __init__(self, spinup_options, time_step=1, df_accuracy_order=2, cache_dirname=None, use_memory_cache=True):
        logger.debug('Initiating {} with cache dirname {}, spinup_options {} time step {}, df_accuracy_order {} and use_memory_cache {}.'.format(self.__class__.__name__, cache_dirname, spinup_options, time_step, df_accuracy_order, use_memory_cache))
        
        ## prepare cache dirname
        if cache_dirname is None:
            cache_dirname = ''
        self.cache_dirname = cache_dirname
        
        ## prepare model
        self.model = Model()
        
        ## prepare time step
        self.time_step = time_step
        
        ## prepare spinup options
        (years, tolerance, combination) = self.model.all_spinup_options(spinup_options)
        if combination == 'and':
            combination = True
        elif combination == 'or':
            combination = False
        else:
            raise ValueError('Combination "{}" unknown.'.format(combination))
        spinup_options = (years, tolerance, combination, df_accuracy_order)
        self.spinup_options = spinup_options
        
        ## prepare memory cache
        if use_memory_cache:
            self.memory_cache = {}
        else:
            self.memory_cache = None
        self.last_parameters = None
    
    
    
    ## access to memory cache
    def load_memory_cache(self, parameters, filename):
        if self.memory_cache is not None and self.last_parameters is not None and np.all(parameters == self.last_parameters):
            try:
                logger.debug('Loading value for {} from memory cache.'.format(filename))
                value = self.memory_cache[filename]
                return value
            except KeyError:
                logger.debug('Value for {} not found in memory cache.'.format(filename))
                return None
        else:
            return None
    
    
    def save_memory_cache(self, parameters, filename, value):
        if self.memory_cache is not None:
            logger.debug('Saving value for {} in memory cache.'.format(filename))
            if self.last_parameters is None or np.any(parameters != self.last_parameters):
                self.last_parameters = parameters
                self.memory_cache = {}
            self.memory_cache[filename] = value
    
    
    
    ## access to cache
    def get_file(self, parameters, filename):
        parameter_set_dir = self.model.get_parameter_set_dir(self.time_step, parameters, create=False)
        
        if parameter_set_dir is not None:
            cache_dir = os.path.join(parameter_set_dir, self.cache_dirname)
            os.makedirs(cache_dir, exist_ok=True)
            file = os.path.join(cache_dir, filename)
        else:
            file = None
        
        return file
    
    
    def load_file(self, parameters, filename, use_memmap=False, as_shared_array=False):
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
    
    
    def save_file(self, parameters, filename, value, save_also_txt=True):
        file = self.get_file(parameters, filename)
        if os.path.exists(file):
            util.io.make_writable(file)
        if save_also_txt:
            logger.debug('Saving value to {} and corresponding text file.'.format(file))
            util.io.save_npy_and_txt(value, file)
        else:
            logger.debug('Saving value to {}.'.format(file))
            util.io.save_npy(value, file)
        util.io.make_read_only(file)
    
    
    def matches_spinup_options(self, parameters, spinup_options_filename):
        needed_options = self.spinup_options
        loaded_options = self.load_file(parameters, spinup_options_filename)
        
        # options = (years, tolerance, combination, df_accuracy_order)
        if loaded_options is not None:
            if needed_options[2]:
                matches = needed_options[0] <= loaded_options[0] and needed_options[1] >= loaded_options[1]
            else:
                matches = needed_options[0] <= loaded_options[0] or needed_options[1] >= loaded_options[1]
            
            if len(loaded_options) == 4:
                matches = matches and needed_options[3] <= loaded_options[3]
        else:
            matches = False
        
        logger.debug('Needed spinup options {} match loaded spinup options {} is {}.'.format(needed_options, loaded_options, matches))
        
        return matches
    
    
    ## value
    def get_value(self, parameters, filename, calculate_function, derivative_used=True, save_also_txt=True, use_memmap=False, as_shared_array=False):
        from .constants import OPTION_FILE_SUFFIX
        
        assert callable(calculate_function)
        
        ## try to load from memory cache
        value = self.load_memory_cache(parameters, filename)
        
        ## if not found try to load from file or calculate
        if value is None:
            filename_root, filename_ext = os.path.splitext(filename)
            option_filename = filename_root + OPTION_FILE_SUFFIX + filename_ext
            
            is_matchig = self.matches_spinup_options(parameters, option_filename)
            
            ## if not matching calculate and save value
            if not is_matchig:
                ## calculating and saving value
                logger.debug('Calculating value with {} and saving with filename {} with derivative_used {}.'.format(calculate_function, filename, derivative_used))
                value = calculate_function(parameters)
                self.save_file(parameters, filename, value, save_also_txt=save_also_txt)
                
                ## saving options
                spinup_options = self.spinup_options
                if not derivative_used:
                    spinup_options = spinup_options[:-1]
                self.save_file(parameters, option_filename, spinup_options, save_also_txt=True)
            
            ## load value if matching or memmap used
            if is_matchig or use_memmap or as_shared_array:
                value = self.load_file(parameters, filename, use_memmap=use_memmap, as_shared_array=as_shared_array)
            
            ## update memory cache
            self.save_memory_cache(parameters, filename, value)
        
        return value

