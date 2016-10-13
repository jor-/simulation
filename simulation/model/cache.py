import os.path

import numpy as np

import measurements.universal.data

import simulation.model.eval
import simulation.model.constants

import util.logging
logger = util.logging.logger



class Cache:

    def __init__(self, model, cache_dirname=None):
        logger.debug('Initiating {} with model {} and cache dirname {}.'.format(self.__class__.__name__, model, cache_dirname))

        self.model = model
        
        if cache_dirname is None:
            cache_dirname = ''
        self.cache_dirname = cache_dirname


    ## option properties

    @property
    def desired_spinup_options(self):
        years = self.model.model_options.spinup_options.years
        tolerance = self.model.model_options.spinup_options.tolerance
        combination = self.model.model_options.spinup_options.combination
        if combination == 'and':
            combination = True
        elif combination == 'or':
            combination = False
        elif not combination in (0, 1):
            raise ValueError('Combination "{}" unknown.'.format(combination))
        spinup_options = (years, tolerance, combination)
        return spinup_options
    
    @property
    def derivative_options(self):
        years = self.model.model_options.derivative_options.years
        step_size = self.model.model_options.derivative_options.step_size
        accuracy_order = self.model.model_options.derivative_options.accuracy_order
        derivative_options = (years, step_size, accuracy_order)
        return derivative_options
    
    @property
    def max_options(self):
        from simulation.model.constants import MODEL_SPINUP_MAX_YEARS
        max_options = (MODEL_SPINUP_MAX_YEARS, 0, False) + self.derivative_options
        return max_options
    
    @property
    def real_spinup_options(self):
        spinup_dir = self.model.spinup_dir
        last_run_dir = self.model.last_run_dir(spinup_dir)
        years = self.model.get_total_years(last_run_dir)
        tolerance = self.model.get_real_tolerance(last_run_dir)
        spinup_options = (years, tolerance, True)
        return spinup_options
    
    @property
    def desired_options(self):
        desired_options = self.desired_spinup_options + self.derivative_options
        assert len(desired_options) == 6
        return desired_options


    ## files

    def get_file(self, filename):
        assert filename is not None
        parameter_set_dir = self.model.parameter_set_dir

        if parameter_set_dir is not None:
            cache_dir = os.path.join(parameter_set_dir, self.cache_dirname)
            file = os.path.join(cache_dir, filename)
        else:
            file = None

        return file
    
    def options_filename(self, filename):
        filename_root, filename_ext = os.path.splitext(filename)
        option_filename = filename_root + simulation.model.constants.DATABASE_CACHE_OPTION_FILE_SUFFIX + '.npy'
        return option_filename
        
    
    ## check options

    def matches_options(self, filename):
        options_filename = self.options_filename(filename)
        loaded_options = self.load_value(options_filename)
        desired_options = self.desired_options

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
    
    def load_value(self, filename, use_memmap=False, as_shared_array=False):
        file = self.get_file(filename)
        if file is not None and os.path.exists(file):
            ## set memmap mode
            if use_memmap or as_shared_array:
                mem_map_mode = 'r'
            else:
                mem_map_mode = None
            ## load
            logger.debug('Loading value from {} with mem_map_mode {} and as_shared_array {}.'.format(file, mem_map_mode, as_shared_array))
            value = util.io.np.load(file, mmap_mode=mem_map_mode)
            ## if scalar, get scalar value
            if value.ndim == 0:
                value = value.reshape(-1)[0]
            ## load as shared array
            elif as_shared_array:
                value = util.parallel.with_multiprocessing.shared_array(value)
        else:
            value = None
        return value


    def _save_value_without_options(self, filename, value, save_also_txt=False):
        ## check input
        if value is None:
            raise ValueError('Value for {} is None!'.format(filename,))
        if filename is None:
            raise ValueError('Filename for is None!')
        
        ## save value
        file = self.get_file(filename)
        assert file is not None
        
        logger.debug('Saving value to {} file with save_also_txt {}.'.format(file, save_also_txt))
        os.makedirs(os.path.dirname(file), exist_ok=True)
        if save_also_txt:
            util.io.np.save_np_and_txt(file, value, make_read_only=True, overwrite=True)
        else:
            util.io.np.save(file, value, make_read_only=True, overwrite=True)


    def save_value(self, filename, value, derivative_used=True, save_also_txt=False):
        ## save value
        self._save_value_without_options(filename, value, save_also_txt=save_also_txt)
        
        ## save option
        options = self.real_spinup_options
        assert len(options) == 3
        if derivative_used:
            options = options + self.derivative_options
            assert len(options) == 6
        
        option_filename = self.options_filename(filename)
        self._save_value_without_options(option_filename, options, save_also_txt=True)

    
    def get_value(self, filename, calculate_function, derivative_used=True, save_also_txt=False, use_memmap=False, as_shared_array=False):
        assert callable(calculate_function)

        ## try to load from file or calculate
        is_matchig = self.matches_options(filename)

        ## if not matching calculate and save value
        if not is_matchig:
            ## calculating and saving value
            logger.debug('Calculating value with {} and saving with filename {} with derivative_used {}.'.format(calculate_function, filename, derivative_used))
            value = calculate_function()
            self.save_value(filename, value, derivative_used=derivative_used, save_also_txt=save_also_txt)

        ## load value if matching or memmap used
        if is_matchig or use_memmap or as_shared_array:
            value = self.load_value(filename, use_memmap=use_memmap, as_shared_array=as_shared_array)

        return value


    def has_value(self, filename):
        return self.matches_options(filename)




class Model_With_F_File_and_MemoryCached(simulation.model.eval.Model_With_F_MemoryCached):
    
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self._cache = Cache(self)
    
    
    def _cached_values_for_boxes(self, time_dim, calculate_function_for_boxes, file_pattern, tracers=None, derivative_used=True):
        assert callable(calculate_function_for_boxes)
        tracers = self.check_tracers(tracers)
    
        ## load cached values from cache
        data_set_name = simulation.model.constants.DATABASE_ALL_DATASET_NAME.format(time_dim=time_dim)

        results_dict = {}
        not_cached_tracers = []
        for tracer in tracers:
            file = file_pattern.format(tracer=tracer, data_set_name=data_set_name)
            if self._cache.has_value(file):
                results_dict[tracer] = self._cache.load_value(file)
            else:
                not_cached_tracers.append(tracer)
        
        ## calculate not cached values
        calculated_results_dict = calculate_function_for_boxes(time_dim, tracers=not_cached_tracers)
        
        ## save calculated values and store in result
        for tracer, tracer_values in calculated_results_dict.items():
            file = file_pattern.format(tracer=tracer, data_set_name=data_set_name)
            self._cache.save_value(file, tracer_values, derivative_used=derivative_used)
            results_dict[tracer] = tracer_values
            
        ## return
        assert (tracers is None and len(results_dict) == self.model_options.tracers_len) or len(results_dict) == len(tracers)
        return results_dict
    

    def f_all(self, time_dim, tracers=None):
        calculate_function_for_boxes = super().f_all
        file_pattern = os.path.join(simulation.model.constants.DATABASE_POINTS_OUTPUT_DIRNAME, simulation.model.constants.DATABASE_F_FILENAME)
        return self._cached_values_for_boxes(time_dim, calculate_function_for_boxes, file_pattern, tracers=tracers, derivative_used=False)

    
    def _cached_values_for_points(self, points, calculate_function_for_points, file_pattern, derivative_used=True):
        ## load cached values and separate not cached points
        not_cached_points_dict = {}
        results_dict = {}
        
        for tracer, tracer_points_dict in points.items():
            results_dict[tracer] = {}
            
            for data_set_name, data_set_points in tracer_points_dict.items():
                file = file_pattern.format(tracer=tracer, data_set_name=data_set_name)
                if self._cache.has_value(file):
                    results_dict[tracer][data_set_name] = self._cache.load_value(file)
                else:
                    try:
                        not_cached_points_dict[tracer]
                    except KeyError:
                        not_cached_points_dict[tracer] = {}
                    not_cached_points_dict[tracer][data_set_name] = data_set_points
        
        ## interpolate not cached values
        calculated_results_dict = calculate_function_for_points(not_cached_points_dict)
        
        ## save interpolated values and store in results dict
        for tracer, tracer_calculated_results_dict in calculated_results_dict.items():
            for data_set_name, data_set_results in tracer_calculated_results_dict.items():
                file = file_pattern.format(tracer=tracer, data_set_name=data_set_name)
                self._cache.save_value(file, data_set_results, derivative_used=derivative_used)
                results_dict[tracer][data_set_name] = data_set_results
        
        ## return
        return results_dict


    def f_points(self, points):       
        calculate_function_for_points = super().f_points 
        file_pattern = os.path.join(simulation.model.constants.DATABASE_POINTS_OUTPUT_DIRNAME, simulation.model.constants.DATABASE_F_FILENAME)
        return self._cached_values_for_points(points, calculate_function_for_points, file_pattern, derivative_used=False)

    
    def _cached_values_for_measurements(self, calculate_function_for_points, *measurements_list):
        ## get base measurements
        not_base_measurements_list = measurements_list
        base_measurements_list = []
        
        while len(not_base_measurements_list) > 0:
            new_not_base_measurements_list = []
            for current_measurements in not_base_measurements_list:
                if isinstance(current_measurements, measurements.universal.data.MeasurementsNearWater):
                    new_not_base_measurements_list.append(current_measurements.base_measurements)
                elif isinstance(current_measurements, measurements.universal.data.MeasurementsAnnualPeriodicUnion) or isinstance(current_measurements, measurements.universal.data.MeasurementsCollection):
                    new_not_base_measurements_list.extend(current_measurements.measurements_list)
                else:
                    base_measurements_list.append(current_measurements)
            not_base_measurements_list = new_not_base_measurements_list
        
        ## calculate results for base measurements (using caching)
        base_measurements_collection = measurements.universal.data.MeasurementsCollection(*base_measurements_list)
        base_points_dict = base_measurements_collection.points_dict
        results_dict = calculate_function_for_points(base_points_dict)
        
        ## convert measurements back if needed
        def convert_back(results_dict, measurements_list):
            for current_measurements in measurements_list:
                if isinstance(current_measurements, measurements.universal.data.MeasurementsNearWater):
                    base_measurements = current_measurements.base_measurements
                    base_results_dict = convert_back(results_dict, [base_measurements])
                    base_results = base_results_dict[base_measurements.tracer][base_measurements.data_set_name]
                    projected_results = current_measurements.near_water_projection_matrix * base_results
                    assert current_measurements.tracer == base_measurements.tracer
                    del results_dict[base_measurements.tracer][base_measurements.data_set_name]
                    results_dict[current_measurements.tracer][current_measurements.data_set_name] = projected_results
                    
                elif isinstance(current_measurements, measurements.universal.data.MeasurementsAnnualPeriodicUnion) or isinstance(current_measurements, measurements.universal.data.MeasurementsCollection):
                    base_measurements_list = current_measurements.measurements_list
                    base_results_dict = convert_back(results_dict, base_measurements_list)
                    base_results_list = [base_results_dict[base_measurements.tracer][base_measurements.data_set_name] for base_measurements in base_measurements_list]
                    projected_results = np.concatenate(base_results_list)
                    for base_measurements in base_measurements_list:
                        assert current_measurements.tracer == base_measurements.tracer
                        del results_dict[base_measurements.tracer][base_measurements.data_set_name]
                    results_dict[current_measurements.tracer][current_measurements.data_set_name] = projected_results
                
            return results_dict

        results_dict = convert_back(results_dict, measurements_list)
        assert set(results_dict.keys()) == {m.tracer for m in measurements_list}
        assert all([set(results_dict[tracer].keys()) == {m.data_set_name for m in measurements_list if m.tracer == tracer} for tracer in results_dict.keys()])
        assert all([len(results_dict[m.tracer][m.data_set_name]) == m.number_of_measurements for m in measurements_list])
        return results_dict


    def f_measurements(self, *measurements_list):
        logger.debug('Calculating f values for measurements {}.'.format(tuple(map(str, measurements_list))))
        return self._cached_values_for_measurements(self.f_points, *measurements_list)




class Model_With_F_And_DF_File_and_MemoryCached(Model_With_F_File_and_MemoryCached, simulation.model.eval.Model_With_F_And_DF_MemoryCached):

    def df_all(self, time_dim, tracers=None, partial_derivative_kind='model_parameters'):
        super_df_all = super().df_all
        calculate_function_for_all = lambda time_dim, tracers: super_df_all(time_dim, tracers=tracers, partial_derivative_kind=partial_derivative_kind)
        file_pattern = os.path.join(simulation.model.constants.DATABASE_POINTS_OUTPUT_DIRNAME, simulation.model.constants.DATABASE_DF_FILENAME.format(derivative_kind=partial_derivative_kind))
        return self._cached_values_for_boxes(time_dim, calculate_function_for_all, file_pattern, tracers=tracers, derivative_used=True)
    

    def df_points(self, points, partial_derivative_kind='model_parameters'):
        super_df_points = super().df_points
        calculate_function_for_points = lambda points: super_df_points(points, partial_derivative_kind=partial_derivative_kind)
        file_pattern = os.path.join(simulation.model.constants.DATABASE_POINTS_OUTPUT_DIRNAME, simulation.model.constants.DATABASE_DF_FILENAME.format(derivative_kind=partial_derivative_kind))
        return self._cached_values_for_points(points, calculate_function_for_points, file_pattern, derivative_used=True)


    def df_measurements(self, *measurements_list, partial_derivative_kind='model_parameters'):
        logger.debug('Calculating df values for measurements {} and partial_derivative_kind {}.'.format(tuple(map(str, measurements_list)), partial_derivative_kind))
        calculate_function_for_points = lambda points: self.df_points(points, partial_derivative_kind=partial_derivative_kind)
        return self._cached_values_for_measurements(calculate_function_for_points, *measurements_list)


Model = Model_With_F_And_DF_File_and_MemoryCached
