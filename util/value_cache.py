import os
import numpy as np

import ndop.model.eval

import util.cache
import util.io.np
import util.io.fs
import util.parallel.with_multiprocessing

import util.logging
logger = util.logging.logger


class MemoryCache:

    def __init__(self, enable=True):
        logger.debug('Initiating {} with enable {}.'.format(self.__class__.__name__, enable))
        self.memory_cache = util.cache.MemoryCache()
        self.last_parameters = None
        self.enable = enable


    def load_value(self, parameters, filename):
        if self.enable and self.last_parameters is not None and np.allclose(parameters, self.last_parameters):
            return self.memory_cache.load_value(filename)
        else:
            raise util.cache.CacheMissError(filename)


    def save_value(self, parameters, filename, value):
        assert value is not None
        if self.enable:
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

    def __init__(self, spinup_options, time_step=1, df_accuracy_order=2, cache_dirname=None, use_memory_cache=True):
        logger.debug('Initiating {} with cache dirname {}, spinup_options {} time step {}, df_accuracy_order {} and use_memory_cache {}.'.format(self.__class__.__name__, cache_dirname, spinup_options, time_step, df_accuracy_order, use_memory_cache))
        from ndop.model.constants import MODEL_SPINUP_MAX_YEARS

        ## prepare cache dirname
        if cache_dirname is None:
            cache_dirname = ''
        self.cache_dirname = cache_dirname

        ## prepare model
        self.model = ndop.model.eval.Model()

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

        self.max_spinup_options = (MODEL_SPINUP_MAX_YEARS, 0, False, df_accuracy_order)

        ## prepare memory cache
        self.memory_cache = MemoryCache(use_memory_cache)
        self.last_parameters = None



    ## access to cache

    def get_parameter_set_dir(self, parameters):
        VALUE_NAME = 'parameter_set_dir'

        try:
            parameter_set_dir = self.memory_cache.load_value(parameters, VALUE_NAME)
        except util.cache.CacheMissError:
            parameter_set_dir = self.model.get_parameter_set_dir(self.time_step, parameters, create=False)
            if parameter_set_dir is not None:
                self.memory_cache.save_value(parameters, VALUE_NAME, parameter_set_dir)

        return parameter_set_dir


    def get_file(self, parameters, filename):
        parameter_set_dir = self.get_parameter_set_dir(parameters)

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
        if value is None:
            raise ValueError('Value for filename {} with parameters {} is None!'.format(filename, parameters))

        file = self.get_file(parameters, filename)
        assert file is not None
        if save_also_txt:
            logger.debug('Saving value to {} and corresponding text file.'.format(file))
            util.io.np.save_npy_and_txt(value, file, make_read_only=True, overwrite=True)
        else:
            logger.debug('Saving value to {}.'.format(file))
            util.io.np.save(file, value, make_read_only=True, overwrite=True)


    def matches_spinup_options(self, parameters, spinup_options_filename):
        loaded_options = self.load_file(parameters, spinup_options_filename)

        def is_matching(needed_options):
            if loaded_options is not None:
                if needed_options[2]:
                    matches = needed_options[0] <= loaded_options[0] and needed_options[1] >= loaded_options[1]
                else:
                    matches = needed_options[0] <= loaded_options[0] or needed_options[1] >= loaded_options[1]

                if len(loaded_options) == 4:
                    matches = matches and needed_options[3] <= loaded_options[3]
            else:
                matches = False
            return matches

        matches_spinup_options = is_matching(self.spinup_options)
        logger.debug('Loaded spinup options {} match needed spinup options {} is {}.'.format(loaded_options, self.spinup_options, matches_spinup_options))

        if not matches_spinup_options:
            matches_spinup_options = is_matching(self.max_spinup_options)
            logger.debug('Loaded spinup options {} match max spinup options {} is {}.'.format(loaded_options, self.max_spinup_options, matches_spinup_options))

        return matches_spinup_options


    ## value
    def get_value(self, parameters, filename, calculate_function, derivative_used=True, save_also_txt=True, use_memmap=False, as_shared_array=False):
        from .constants import OPTION_FILE_SUFFIX

        assert callable(calculate_function)

        ## try to load from memory cache
        try:
            value = self.memory_cache.load_value(parameters, filename)

        ## if not found try to load from file or calculate
        except util.cache.CacheMissError:
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
            self.memory_cache.save_value(parameters, filename, value)

        return value


    def has_value(self, parameters, filename):
        # file = self.get_file(parameters, filename)
        # return file is not None and os.path.exists(file)
        from .constants import OPTION_FILE_SUFFIX
        filename_root, filename_ext = os.path.splitext(filename)
        option_filename = filename_root + OPTION_FILE_SUFFIX + filename_ext
        return self.matches_spinup_options(parameters, option_filename)






# class Cache:
#
#     def __init__(self, spinup_options, time_step=1, df_accuracy_order=2, cache_dirname=None, use_memory_cache=True):
#         logger.debug('Initiating {} with cache dirname {}, spinup_options {} time step {}, df_accuracy_order {} and use_memory_cache {}.'.format(self.__class__.__name__, cache_dirname, spinup_options, time_step, df_accuracy_order, use_memory_cache))
#
#         ## prepare cache dirname
#         if cache_dirname is None:
#             cache_dirname = ''
#         self.cache_dirname = cache_dirname
#
#         ## prepare model
#         self.model = ndop.model.eval.Model()
#
#         ## prepare time step
#         self.time_step = time_step
#
#         ## prepare spinup options
#         (years, tolerance, combination) = self.model.all_spinup_options(spinup_options)
#         if combination == 'and':
#             combination = True
#         elif combination == 'or':
#             combination = False
#         else:
#             raise ValueError('Combination "{}" unknown.'.format(combination))
#         spinup_options = (years, tolerance, combination, df_accuracy_order)
#         self.spinup_options = spinup_options
#
#         ## prepare memory cache
#         if use_memory_cache:
#             # self.memory_cache = {}
#             self.memory_cache = util.cache.MemoryCache()
#         else:
#             self.memory_cache = None
#         self.last_parameters = None
#
#
#
#     ## access to memory cache
#     def load_memory_cache(self, parameters, filename):
#         if self.memory_cache is not None and self.last_parameters is not None and np.allclose(parameters, self.last_parameters):
#             # try:
#             #     logger.debug('Loading value for {} from memory cache.'.format(filename))
#             #     return self.memory_cache[filename]
#             # except KeyError:
#             #     logger.debug('Value for {} not found in memory cache.'.format(filename))
#             #     raise util.cache.CacheMissError(filename)
#             return self.memory_cache.load_value(filename)
#         else:
#             raise util.cache.CacheMissError(filename)
#
#
#     def save_memory_cache(self, parameters, filename, value):
#         assert value is not None
#
#         if self.memory_cache is not None:
#             logger.debug('Saving value for {} in memory cache.'.format(filename))
#             if self.last_parameters is None or np.any(parameters != self.last_parameters):
#                 self.last_parameters = parameters
#                 # self.memory_cache = {}
#                 self.memory_cache = util.cache.MemoryCache()
#             # self.memory_cache[filename] = value
#             self.memory_cache.save_value(filename, value)
#
#
#     def get_value_memory_cache(self, parameters, filename, calculate_function, derivative_used=True, save_also_txt=True, use_memmap=False, as_shared_array=False):
#         from .constants import OPTION_FILE_SUFFIX
#
#         assert callable(calculate_function)
#
#         ## try to load from memory cache
#         try:
#             value = self.load_memory_cache(parameters, filename)
#
#         ## if not found try to load from file or calculate
#         except util.cache.CacheMissError:
#             filename_root, filename_ext = os.path.splitext(filename)
#             option_filename = filename_root + OPTION_FILE_SUFFIX + filename_ext
#
#             is_matchig = self.matches_spinup_options(parameters, option_filename)
#
#             ## if not matching calculate and save value
#             if not is_matchig:
#                 ## calculating and saving value
#                 logger.debug('Calculating value with {} and saving with filename {} with derivative_used {}.'.format(calculate_function, filename, derivative_used))
#                 value = calculate_function(parameters)
#                 self.save_file(parameters, filename, value, save_also_txt=save_also_txt)
#
#                 ## saving options
#                 spinup_options = self.spinup_options
#                 if not derivative_used:
#                     spinup_options = spinup_options[:-1]
#                 self.save_file(parameters, option_filename, spinup_options, save_also_txt=True)
#
#             ## load value if matching or memmap used
#             if is_matchig or use_memmap or as_shared_array:
#                 value = self.load_file(parameters, filename, use_memmap=use_memmap, as_shared_array=as_shared_array)
#
#             ## update memory cache
#             self.save_memory_cache(parameters, filename, value)
#
#         return value
#
#
#
#     ## access to cache
#
#     def get_parameter_set_dir(self, parameters):
#         VALUE_NAME = 'parameter_set_dir'
#
#         try:
#             parameter_set_dir = self.load_memory_cache(parameters, VALUE_NAME)
#         except util.cache.CacheMissError:
#             parameter_set_dir = self.model.get_parameter_set_dir(self.time_step, parameters, create=False)
#             if parameter_set_dir is not None:
#                 self.save_memory_cache(parameters, VALUE_NAME, parameter_set_dir)
#
#         return parameter_set_dir
#
#
#     def get_file(self, parameters, filename):
#         parameter_set_dir = self.get_parameter_set_dir(parameters)
#
#         if parameter_set_dir is not None:
#             cache_dir = os.path.join(parameter_set_dir, self.cache_dirname)
#             os.makedirs(cache_dir, exist_ok=True)
#             file = os.path.join(cache_dir, filename)
#         else:
#             file = None
#
#         return file
#
#
#     def load_file(self, parameters, filename, use_memmap=False, as_shared_array=False):
#         file = self.get_file(parameters, filename)
#         if file is not None and os.path.exists(file):
#             if use_memmap or as_shared_array:
#                 mem_map_mode = 'r'
#             else:
#                 mem_map_mode = None
#             logger.debug('Loading value from {} with mem_map_mode {} and as_shared_array {}.'.format(file, mem_map_mode, as_shared_array))
#             value = np.load(file, mmap_mode=mem_map_mode)
#             if as_shared_array:
#                 value = util.parallel.with_multiprocessing.shared_array(value)
#         else:
#             value = None
#         return value
#
#
#
#     def save_file(self, parameters, filename, value, save_also_txt=True):
#         if value is None:
#             raise ValueError('Value for filename {} with parameters {} is None!'.format(filename, parameters))
#
#         file = self.get_file(parameters, filename)
#         assert file is not None
#         if save_also_txt:
#             logger.debug('Saving value to {} and corresponding text file.'.format(file))
#             util.io.np.save_npy_and_txt(value, file, make_read_only=True, overwrite=True)
#         else:
#             logger.debug('Saving value to {}.'.format(file))
#             util.io.np.save(file, value, make_read_only=True, overwrite=True)
#
#
#     def matches_spinup_options(self, parameters, spinup_options_filename):
#         needed_options = self.spinup_options
#         loaded_options = self.load_file(parameters, spinup_options_filename)
#
#         # options = (years, tolerance, combination, df_accuracy_order)
#         if loaded_options is not None:
#             if needed_options[2]:
#                 matches = needed_options[0] <= loaded_options[0] and needed_options[1] >= loaded_options[1]
#             else:
#                 matches = needed_options[0] <= loaded_options[0] or needed_options[1] >= loaded_options[1]
#
#             if len(loaded_options) == 4:
#                 matches = matches and needed_options[3] <= loaded_options[3]
#         else:
#             matches = False
#
#         logger.debug('Needed spinup options {} match loaded spinup options {} is {}.'.format(needed_options, loaded_options, matches))
#
#         return matches
#
#
#     ## value
#     def get_value(self, parameters, filename, calculate_function, derivative_used=True, save_also_txt=True, use_memmap=False, as_shared_array=False):
#         from .constants import OPTION_FILE_SUFFIX
#
#         assert callable(calculate_function)
#
#         ## try to load from memory cache
#         try:
#             value = self.load_memory_cache(parameters, filename)
#
#         ## if not found try to load from file or calculate
#         except util.cache.CacheMissError:
#             filename_root, filename_ext = os.path.splitext(filename)
#             option_filename = filename_root + OPTION_FILE_SUFFIX + filename_ext
#
#             is_matchig = self.matches_spinup_options(parameters, option_filename)
#
#             ## if not matching calculate and save value
#             if not is_matchig:
#                 ## calculating and saving value
#                 logger.debug('Calculating value with {} and saving with filename {} with derivative_used {}.'.format(calculate_function, filename, derivative_used))
#                 value = calculate_function(parameters)
#                 self.save_file(parameters, filename, value, save_also_txt=save_also_txt)
#
#                 ## saving options
#                 spinup_options = self.spinup_options
#                 if not derivative_used:
#                     spinup_options = spinup_options[:-1]
#                 self.save_file(parameters, option_filename, spinup_options, save_also_txt=True)
#
#             ## load value if matching or memmap used
#             if is_matchig or use_memmap or as_shared_array:
#                 value = self.load_file(parameters, filename, use_memmap=use_memmap, as_shared_array=as_shared_array)
#
#             ## update memory cache
#             self.save_memory_cache(parameters, filename, value)
#
#         return value
#
#
#     def has_value(self, parameters, filename):
#         file = self.get_file(parameters, filename)
#         return file is not None and os.path.exists(file)

