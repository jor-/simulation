import os
import logging
import numpy as np

from ndop.model.eval import Model

import util.io



class Cache:
    
    def __init__(self, spinup_options, time_step=1, df_accuracy_order=2, cache_dirname=None):
        logging.debug('Initiating {} with cache dirname {} and time step {}.'.format(self.__class__.__name__, cache_dirname, time_step))
        
        if cache_dirname is None:
            cache_dirname = ''
        self.cache_dirname = cache_dirname
        
        self.model = Model()
        
        self.time_step = time_step
        
        (years, tolerance, combination) = self.model.all_spinup_options(spinup_options)
        if combination == 'and':
            combination = True
        elif combination == 'or':
            combination = False
        else:
            raise ValueError('Combination "{}" unknown.'.format(combination))
        spinup_options = (years, tolerance, combination, df_accuracy_order)
        self.spinup_options = spinup_options
    
    
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
    
    def load_file(self, parameters, filename):
        file = self.get_file(parameters, filename)
        if file is not None and os.path.exists(file):
            values = np.load(file)
            logging.debug('Got values from {}.'.format(file))
        else:
            values = None
        return values
    
    def save_file(self, parameters, filename, values, save_also_txt=True):
        file = self.get_file(parameters, filename)
        if save_also_txt:
            logging.debug('Saving value to {}.'.format(file))
            util.io.save_npy_and_txt(values, file)
        else:
            logging.debug('Saving value to {} and corresponding text file.'.format(file))
            util.io.save_npy(values, file)
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
        
        logging.debug('Needed spinup options {} match loaded spinup options {} is {}.'.format(needed_options, loaded_options, matches))
        
        return matches
    
    
    ## value
    def get_value(self, parameters, filename, calculate_function, derivative_used=True, save_also_txt=True):
        from .constants import OPTION_FILE_SUFFIX
        
        assert callable(calculate_function)
        
        filename_root, filename_ext = os.path.splitext(filename)
        option_filename = filename_root + OPTION_FILE_SUFFIX + filename_ext
        
        ## if matching load value
        if self.matches_spinup_options(parameters, option_filename):
            logging.debug('Value loading from {}.'.format(filename))
            value = self.load_file(parameters, filename)
        ## else calculate and save value
        else:
            logging.debug('Calculating value with {} and saving with filename {} with derivative_used {}.'.format(calculate_function, filename, derivative_used))
            value = calculate_function(parameters)
            self.save_file(parameters, filename, value, save_also_txt=save_also_txt)
            spinup_options = self.spinup_options
            if not derivative_used:
                spinup_options = spinup_options[:-1]
            self.save_file(parameters, option_filename, spinup_options, save_also_txt=True)
        
        return value

