import os.path

import numpy as np

import measurements.universal.data

import simulation.model.eval
import simulation.model.constants

import util.logging


class Cache:

    def __init__(self, model, cache_dirname=None):
        util.logging.debug(f'Initiating {self.__class__.__name__} with model {model} and cache dirname {cache_dirname}.')
        self.model = model
        if cache_dirname is None:
            cache_dirname = ''
        self.cache_dirname = cache_dirname

    # *** file *** #

    def get_file(self, filename, derivative_used, **filename_format_dict):
        assert filename is not None

        model = self.model

        if model.is_matching_run_available:
            real_years = model.real_years()
            filename = filename.format(spinup_years=real_years, **filename_format_dict)
            bottom_dirs, filename = os.path.split(filename)

            if derivative_used:
                derivative_dirname = simulation.model.constants.DATABASE_CACHE_DERIVATIVE_DIRNAME.format(derivative_step_size=self.model.model_options.derivative_options.step_size, derivative_years=self.model.model_options.derivative_options.years, derivative_accuracy_order=self.model.model_options.derivative_options.accuracy_order)
                bottom_dirs = os.path.join(bottom_dirs, derivative_dirname)

            file = os.path.join(model.parameter_set_dir, self.cache_dirname, bottom_dirs, filename)
        else:
            file = None

        return file

    # *** value *** #

    def has_value(self, file):
        return file is not None and os.path.exists(file)

    def load_value(self, file, use_memmap=False, as_shared_array=False):
        if file is not None and os.path.exists(file):
            # set memmap mode
            if use_memmap or as_shared_array:
                mem_map_mode = 'r'
            else:
                mem_map_mode = None
            # load
            util.logging.debug(f'Loading value from {file} with mem_map_mode {mem_map_mode} and as_shared_array {as_shared_array}.')
            value = util.io.np.load_np_or_txt(file, mmap_mode=mem_map_mode)
            # if scalar, get scalar value
            if value.ndim == 0:
                value = value.reshape(-1)[0]
            # load as shared array
            elif as_shared_array:
                value = util.parallel.with_multiprocessing.shared_array(value)
        else:
            value = None
        return value

    def save_value(self, file, value, save_as_np=True, save_as_txt=False):
        # check input
        if value is None:
            raise ValueError(f'Value for {file} is None!')
        if file is None:
            raise ValueError('File is None!')

        # save value
        util.logging.debug(f'Saving value to {file} with save_as_np {save_as_np} and save_as_txt {save_as_txt}.')
        os.makedirs(os.path.dirname(file), exist_ok=True)
        try:
            util.io.np.save_np_or_txt(file, value, make_read_only=True, overwrite=False, save_as_np=save_as_np, save_as_txt=save_as_txt)
        except PermissionError as e:
            if os.path.exists(file):
                saved_value = util.io.np.load_np_or_txt(file)
                if not np.allclose(value, saved_value):
                    raise e
            else:
                raise e

    def get_value(self, file, calculate_function, save_as_np=True, save_as_txt=False, use_memmap=False, as_shared_array=False):
        assert callable(calculate_function)

        # if not matching calculate and save value
        is_matchig = self.has_value(file)
        if not is_matchig:

            # calculating and saving value
            util.logging.debug(f'Calculating value with {calculate_function} and saving with file {file}.')
            value = calculate_function()
            self.save_value(file, value, save_as_np=save_as_np, save_as_txt=save_as_txt)

        # load value if matching or memmap used
        if is_matchig or use_memmap or as_shared_array:
            value = self.load_value(file, use_memmap=use_memmap, as_shared_array=as_shared_array)

        return value


class Model_With_F_File_and_MemoryCached(simulation.model.eval.Model_With_F_MemoryCached):

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self._cache = Cache(self)

    def _cached_values_for_boxes(self, time_dim, calculate_function_for_boxes, file_pattern, derivative_used, tracers=None):
        assert callable(calculate_function_for_boxes)
        tracers = self.check_tracers(tracers)

        # load cached values from cache
        data_set_name = simulation.model.constants.DATABASE_ALL_DATASET_NAME.format(time_dim=time_dim)

        results_dict = {}
        not_cached_tracers = []
        for tracer in tracers:
            file = self._cache.get_file(file_pattern, derivative_used=derivative_used, tracer=tracer, data_set_name=data_set_name)
            if self._cache.has_value(file):
                results_dict[tracer] = self._cache.load_value(file)
            else:
                not_cached_tracers.append(tracer)

        # calculate not cached values
        calculated_results_dict = calculate_function_for_boxes(time_dim, tracers=not_cached_tracers)

        # save calculated values and store in result
        for tracer, tracer_values in calculated_results_dict.items():
            file = self._cache.get_file(file_pattern, derivative_used=derivative_used, tracer=tracer, data_set_name=data_set_name)
            self._cache.save_value(file, tracer_values)
            results_dict[tracer] = tracer_values

        # return
        assert (tracers is None and len(results_dict) == self.model_options.tracers_len) or len(results_dict) == len(tracers)
        return results_dict

    def f_all(self, time_dim, tracers=None):
        calculate_function_for_boxes = super().f_all
        file_pattern = os.path.join(simulation.model.constants.DATABASE_POINTS_OUTPUT_DIRNAME, simulation.model.constants.DATABASE_F_FILENAME)
        return self._cached_values_for_boxes(time_dim, calculate_function_for_boxes, file_pattern, derivative_used=False, tracers=tracers)

    def _cached_values_for_points(self, points, calculate_function_for_points, file_pattern, derivative_used):
        # load cached values and separate not cached points
        not_cached_points_dict = {}
        results_dict = {}

        for tracer, tracer_points_dict in points.items():
            results_dict[tracer] = {}

            for data_set_name, data_set_points in tracer_points_dict.items():
                file = self._cache.get_file(file_pattern, derivative_used=derivative_used, tracer=tracer, data_set_name=data_set_name)
                if self._cache.has_value(file):
                    results_dict[tracer][data_set_name] = self._cache.load_value(file)
                else:
                    try:
                        not_cached_points_dict[tracer]
                    except KeyError:
                        not_cached_points_dict[tracer] = {}
                    not_cached_points_dict[tracer][data_set_name] = data_set_points

        # interpolate not cached values
        calculated_results_dict = calculate_function_for_points(not_cached_points_dict)

        # save interpolated values and store in results dict
        for tracer, tracer_calculated_results_dict in calculated_results_dict.items():
            for data_set_name, data_set_results in tracer_calculated_results_dict.items():
                file = self._cache.get_file(file_pattern, derivative_used=derivative_used, tracer=tracer, data_set_name=data_set_name)
                self._cache.save_value(file, data_set_results)
                results_dict[tracer][data_set_name] = data_set_results

        # return
        return results_dict

    def f_points(self, points):
        calculate_function_for_points = super().f_points
        file_pattern = os.path.join(simulation.model.constants.DATABASE_POINTS_OUTPUT_DIRNAME, simulation.model.constants.DATABASE_F_FILENAME)
        return self._cached_values_for_points(points, calculate_function_for_points, file_pattern, derivative_used=False)

    def _cached_values_for_measurements(self, calculate_function_for_points, *measurements_list):
        # get base measurements
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

        # calculate results for base measurements (using caching)
        base_measurements_collection = measurements.universal.data.MeasurementsCollection(*base_measurements_list)
        base_points_dict = base_measurements_collection.points_dict
        results_dict = calculate_function_for_points(base_points_dict)

        # convert measurements back if needed
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
        util.logging.debug(f'Calculating f values for measurements {tuple(map(str, measurements_list))}.')
        return self._cached_values_for_measurements(self.f_points, *measurements_list)


class Model_With_F_And_DF_File_and_MemoryCached(Model_With_F_File_and_MemoryCached, simulation.model.eval.Model_With_F_And_DF_MemoryCached):

    def df_all(self, time_dim, tracers=None, partial_derivative_kind='model_parameters'):
        super_df_all = super().df_all

        def calculate_function_for_all(time_dim, tracers):
            return super_df_all(time_dim, tracers=tracers, partial_derivative_kind=partial_derivative_kind)

        file_pattern = os.path.join(simulation.model.constants.DATABASE_POINTS_OUTPUT_DIRNAME, simulation.model.constants.DATABASE_DF_FILENAME.format(derivative_kind=partial_derivative_kind))
        return self._cached_values_for_boxes(time_dim, calculate_function_for_all, file_pattern, derivative_used=True, tracers=tracers)

    def df_points(self, points, partial_derivative_kind='model_parameters'):
        super_df_points = super().df_points

        def calculate_function_for_points(points):
            return super_df_points(points, partial_derivative_kind=partial_derivative_kind)

        file_pattern = os.path.join(simulation.model.constants.DATABASE_POINTS_OUTPUT_DIRNAME, simulation.model.constants.DATABASE_DF_FILENAME.format(derivative_kind=partial_derivative_kind))
        return self._cached_values_for_points(points, calculate_function_for_points, file_pattern, derivative_used=True)

    def df_measurements(self, *measurements_list, partial_derivative_kind='model_parameters'):
        util.logging.debug(f'Calculating df values for measurements {tuple(map(str, measurements_list))} and partial_derivative_kind {partial_derivative_kind}.')

        def calculate_function_for_points(points):
            return self.df_points(points, partial_derivative_kind=partial_derivative_kind)
        return self._cached_values_for_measurements(calculate_function_for_points, *measurements_list)


Model = Model_With_F_And_DF_File_and_MemoryCached
