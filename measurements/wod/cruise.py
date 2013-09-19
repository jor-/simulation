import scipy.io
import numpy as np

import multiprocessing
import datetime
import warnings
import itertools
import math

import ndop.metos3d.data

import util.io
import util.datetime
from util.debug import Debug, print_debug

from .constants import CRUISES_PICKLED_FILE, MEASUREMENTS_PICKLED_FILE, DATA_DIR

class Cruise(Debug):
    
    def __init__(self, file, debug_level=0, required_debug_level=1):
        from . import constants
        
        Debug.__init__(self, debug_level, required_debug_level-1, 'ndop.measurements.wod.cruise.Cruise: ')
        
        self.print_debug_inc(('Loading cruise from ', file, '.'))
        
        ## open netcdf file
        f = scipy.io.netcdf.netcdf_file(file, 'r')
        
        ## read time and data
        day_offset = float(f.variables[constants.DAY_OFFSET].data)
        hours_offset = (day_offset % 1) * 24
        minutes_offset = (hours_offset % 1) * 60
        seconds_offset = (minutes_offset % 1) * 60
        
        day_offset = int(day_offset)
        hours_offset = int(hours_offset)
        minutes_offset = int(minutes_offset)
        seconds_offset = int(seconds_offset)
        
        dt_offset = datetime.timedelta(days=day_offset, hours=hours_offset, minutes=minutes_offset, seconds=seconds_offset)
        dt = constants.BASE_DATE + dt_offset
        dt_float = util.datetime.datetime_to_float(dt)
        
        self.dt_float = dt_float
        
        ## read coordinates and valid measurements
        self.x = float(f.variables[constants.LON].data)
        self.y = float(f.variables[constants.LAT].data)
        
        z_flag = f.variables[constants.DEPTH_FLAG].data
        po4_flag = f.variables[constants.PO4_FLAG].data
        po4_profile_flag = f.variables[constants.PO4_PROFILE_FLAG].data
        valid_mask = np.logical_and(po4_flag == 0, z_flag == 0) * (po4_profile_flag == 0)
        
        z = f.variables[constants.DEPTH].data[valid_mask]
        po4 = f.variables[constants.PO4].data[valid_mask]
        
        valid_mask = po4 != constants.MISSING_VALUE
        z = z[valid_mask]
        po4 = po4[valid_mask]
        
        self.z = z
        self.po4 = po4
        
        ## close file
        f.close()
        
        ## check values
        if np.any(po4 < 0):
            warnings.warn('PO4 in ' + file + ' is lower then 0.')
            valid_mask = po4 > 0
            po4 = po4[valid_mask]
            z = z[valid_mask]
        
        if np.any(z < 0):
            warnings.warn('Depth in ' + file + ' is lower then 0.')
            z[z < 0] = 0
        
        self.print_debug_dec(('Cruise from ', file, ' loaded.'))
    
    @property
    def number_of_measurements(self):
        return self.po4.size
    
    @property
    def land_sea_mask(self):
        try:
            return self.__land_sea_mask
        except AttributeError:
            raise Exception('Land sea mask is not set.')
    
    @land_sea_mask.setter
    def land_sea_mask(self, land_sea_mask):
        self.__land_sea_mask = land_sea_mask
        self.__spatial_indices = None
    
    @property
    def spatial_indices(self):
        try:
            indices = self.__spatial_indices
        except AttributeError:
            indices = None
        
        if indices == None:
            self.print_debug_inc_dec(('Calculating spatil indices.'))
            
            land_sea_mask = self.land_sea_mask
            x = self.x
            y = self.y
            z = self.z
            
            m = z.size
            
            indices = np.empty((m, 3), dtype=np.uint16)
            
            for i in range(m):
                indices[i] = ndop.metos3d.data.get_spatial_index(x, y, z[i], land_sea_mask, debug_level=self.debug_level, required_debug_level=self.required_debug_level+1)
            
            self.__spatial_indices = indices
        
        return indices
    
    @spatial_indices.setter
    def spatial_indices(self, spatial_indices):
        self.__spatial_indices = spatial_indices
    
    
    @property
    def year(self):
        year = int(self.dt_float)
        return year
    
    @property
    def year_fraction(self):
        year_fraction = self.dt_float % 1
        return year_fraction
    
    def is_year_fraction_in(self, lower_bound=float('-inf'), upper_bound=float('inf')):
        year_fraction = self.year_fraction
        return year_fraction >= lower_bound and year_fraction < upper_bound





class Cruise_Collection(Debug):
    
    def __init__(self, cruises=None, debug_level=0, required_debug_level=1):
        Debug.__init__(self, debug_level, required_debug_level-1, 'ndop.measurements.wod.cruise.Cruise_Collection: ')
        self.__cruises = cruises
    
    
    @property
    def cruises(self):
        try:
            cruises = self.__cruises
        except AttributeError:
            cruises = None
        
        if cruises == None:
            try:
                self.load_cruises_from_pickle_file()
            except (OSError, IOError):
                self.load_cruises_from_netcdf()
                self.save_cruises_to_pickle_file()
            
            cruises = self.cruises
        
        return cruises
    
    @cruises.setter
    def cruises(self, cruises):
        self.__cruises = cruises
    
    
    def calculate_spatial_indices(self):
        cruises = self.cruises
        
        self.print_debug_inc(('Calculating spatial indices for ', len(cruises), ' cruises.'))
        
        land_sea_mask = ndop.metos3d.data.load_land_sea_mask(self.debug_level, self.required_debug_level+1)
        
        for cruise in cruises:
            cruise.land_sea_mask = land_sea_mask
            cruise.spatial_indices
            
        self.print_debug_dec(('For ', len(cruises), ' cruises spatial indices calculted.'))
    
    
    def load_cruises_from_netcdf(self, data_dir=DATA_DIR):
        self.print_debug_inc('Loading all cruises from netcdf files.')
        
        ## lookup files
        self.print_debug(('Looking up files in ', data_dir, '.'))
        files = util.io.get_files(data_dir)
        self.print_debug((len(files), ' files found.'))
        
        ## load land sea mask
        land_sea_mask = ndop.metos3d.data.load_land_sea_mask(self.debug_level, self.required_debug_level+1)
        
        ## load cruises
        self.print_debug('Loading cruises from found files.')
        def load_cruise(file):
            cruise = Cruise(file, self.debug_level, self.required_debug_level+1)
            cruise.land_sea_mask = land_sea_mask
            cruise.spatial_indices
            return cruise
        
        cruises = [load_cruise(file) for file in files]
        self.print_debug((len(cruises), ' cruises loaded.'))
        
        ## remove empty cruises
        self.print_debug('Removing empty cruises.')
        cruises = [cruise for cruise in cruises if cruise.number_of_measurements > 0]
        self.print_debug_dec((len(cruises), ' not empty cruises found.'))
        
        ## return cruises
        self.cruises = cruises
    
    
    
    
    def save_cruises_to_pickle_file(self, file=CRUISES_PICKLED_FILE):
        self.print_debug_inc(('Saving cruises at ', file))
        util.io.save_object(self.cruises, file)
        self.print_debug_dec(('Cruises saved at ', file))
    
    
    def load_cruises_from_pickle_file(self, file=CRUISES_PICKLED_FILE):
        self.print_debug_inc(('Loading cruises at ', file))
        self.cruises = util.io.load_object(file)
        self.print_debug_dec(('Cruises loaded from ', file))
        
    
    

class Measurements(Debug):
    import ndop.measurements.util
    
    def __init__(self, debug_level=0, required_debug_level=1):
        Debug.__init__(self, debug_level, required_debug_level-1, 'ndop.measurements.wod.cruise.Measurements: ')
        self.measurements_dict = dict()
    
    
    def add_cruises_with_box_indices(self, cruises):
        measurements_dict = self.measurements_dict
        
        ## insert results in dict
        for cruise in cruises:
            cruise_spatial_indices = cruise.spatial_indices
            cruise_dt_float = cruise.dt_float
            cruise_results = cruise.po4
            
            for i in range(cruise_results.size):
                space_index = tuple(cruise_spatial_indices[i].tolist())
                
                # get value if in dict or new dict if not in dict
                time_dict = measurements_dict.setdefault(space_index, dict())
                result_list = time_dict.setdefault(cruise_dt_float, [])
                result_list.append(cruise_results[i].astype(float))
    
    
    def add_cruises_with_coordinates(self, cruises):
        measurements_dict = self.measurements_dict
        
        ## insert results in dict
        for cruise in cruises:
            cruise_x = cruise.x
            cruise_y = cruise.y
            cruise_z = cruise.z
            cruise_dt_float = cruise.dt_float
            cruise_results = cruise.po4
            
            for i in range(cruise_results.size):
                space_index = (cruise_x, cruise_y, cruise_z[i])
                time_dict = measurements_dict.setdefault(space_index, dict())
                result_list = time_dict.setdefault(cruise_dt_float, [])
                result_list.append(cruise_results[i].astype(float))
    
    
    
    def save_to_pickle_file(self, file=MEASUREMENTS_PICKLED_FILE):
        self.print_debug_inc(('Saving measurements at ', file))
        util.io.save_object(self.measurements_dict, file)
        self.print_debug_dec(('Measurements saved at ', file))
    
    
    def load_from_pickle_file(self, file=MEASUREMENTS_PICKLED_FILE):
        self.print_debug_inc(('Loading measurements at ', file))
        self.measurements_dict = util.io.load_object(file)
        self.print_debug_dec(('Measurements loaded from ', file))
    
    
    
    def transform_indices(self, transform_function):
        measurements_dict = self.measurements_dict
        transformed_measurements_dict = dict()
        
        for (space_index, time_dict) in measurements_dict.items():
            
            for (time_index, results_list) in time_dict.items():
                index = list(space_index) + [time_index]
                transformed_index = transform_function(index)
                transformed_space_index = transformed_index[0:-1]
                transformed_time_index = transformed_index[-1]
                
                transformed_time_dict = transformed_measurements_dict.setdefault(transformed_space_index, dict())
                transformed_result_list = transformed_time_dict.setdefault(transformed_time_index, [])
                transformed_result_list += results_list
        
        transformed_measurements = Measurements(debug_level=self.debug_level, required_debug_level=self.required_debug_level)
        transformed_measurements.measurements_dict = transformed_measurements_dict
        
        return transformed_measurements
    
    
    def categorize_indices(self, separation_values):
        def transform_function(index):
            for i in range(len(separation_values)):
                if separation_values[i] is not None:
                     index[i] = math.floor(index[i] / separation_values[i]) * separation_values[i]
            
            return index
            
        transformed_measurements = self.transform_indices(transform_function)
        return transformed_measurements
    
#     def categorize_time(self, separation_value):
#         def transform_function(index):
#             index[3] = math.floor(index[3] / separation_value) * separation_value
#             return index
#             
#         transformed_measurements = self.transform_indices(transform_function)
#         return transformed_measurements
    
    
    def discard_year(self):
        def transform_function(index):
            index[3] = index[3] % 1
            return index
            
        transformed_measurements = self.transform_indices(transform_function)
        return transformed_measurements
    
    
#     def transform_time(self, transform_function):
#         measurements_dict = self.measurements_dict
#         transformed_measurements_dict = dict()
#         
#         for (space_index, time_dict) in measurements_dict.items():
#             transformed_time_dict = transformed_measurements_dict.setdefault(space_index, dict())
#             
#             for (time_index, results_list) in time_dict.items():
#                 transformed_time_index = transform_function(time_index)
#                 transformed_result_list = transformed_time_dict.setdefault(transformed_time_index, [])
#                 transformed_result_list += results_list
#         
#         transformed_measurements = Measurements(debug_level=self.debug_level, required_debug_level=self.required_debug_level)
#         transformed_measurements.measurements_dict = transformed_measurements_dict
#         
#         return transformed_measurements
#     
#     
#     def categorize_time(self, separation_value):
#         transform_function = lambda t: math.floor(t / separation_value) * separation_value
#         transformed_measurements = self.transform_time(transform_function)
#         
#         return transformed_measurements
#     
#     def discard_year(self):
#         transform_function = lambda t: t%1
#         transformed_measurements = self.transform_time(transform_function)
#         
#         return transformed_measurements
    
    
    
    def filter_by_space(self, filter_function):
        measurements_dict = self.measurements_dict
        filtered_measurements_dict = dict()
        
        for (space_index, time_dict) in measurements_dict.items():
            if filter_function(space_index):
                filtered_measurements_dict[space_index] = time_dict
        
        filtered_measurements = Measurements(debug_level=self.debug_level, required_debug_level=self.required_debug_level)
        filtered_measurements.measurements_dict = filtered_measurements_dict
        
        return filtered_measurements
    
    
    def filter_space_values(self, x=None, y=None, z=None):
        compare_value_function = lambda value1, value2: value1 is None or value2 is None or value1 == value2 
        filter_function = lambda space_index: compare_value_function(x, space_index[0]) and compare_value_function(y, space_index[1]) and compare_value_function(z, space_index[2])
        
        return self.filter_by_space(filter_function)
    
    
    def filter_by_time(self, filter_function):
        measurements_dict = self.measurements_dict
        filtered_measurements_dict = dict()
        
        for (space_index, time_dict) in measurements_dict.items():
            for (time_index, results_list) in time_dict.items():
                if filter_function(time_index):
                    time_dict = filtered_measurements_dict.setdefault(space_index, dict())
                    time_dict[time_index] = results_list
        
        filtered_measurements = Measurements(debug_level=self.debug_level, required_debug_level=self.required_debug_level)
        filtered_measurements.measurements_dict = filtered_measurements_dict
        
        return filtered_measurements
    
    
    def filter_time_range(self, lower_bound, upper_bound):
        filter_function = lambda time: time >= lower_bound and time <= upper_bound
        
        return self.filter_by_time(filter_function)
    
    
    
    
    def iterate_space_discard_time(self, fun, minimum_measurements=1):
        measurements_dict = self.measurements_dict
#         spatial_indices_list = []
        value_list = []
        
        for (space_index, time_dict) in measurements_dict.items():
            results_list = list(itertools.chain(*time_dict.values()))
            
            if len(results_list) >= minimum_measurements:
#                 spatial_indices_list.append(space_index)
                
                results = np.array(results_list)
                value = fun(results)
                
                row = space_index + (value,)
                
                value_list.append(row)
        
#         spatial_indices = np.array(spatial_indices_list, dtype=np.uint16)
        values = np.array(value_list)
        
#         return (spatial_indices, values)
        return values
    
#     def discard_axis(self, axis=()):
        
    
    def iterate_space_time(self, fun, minimum_measurements=1):
        measurements_dict = self.measurements_dict
        value_list = []
        
        for (space_index, time_dict) in measurements_dict.items():
            for (time_index, results_list) in time_dict.items():
                if len(results_list) >= minimum_measurements:
                    index = (time_index,) + space_index
                    results = np.array(results_list)
                    value = fun(results)
                    row = index + (value,)
                    
                    value_list.append(row)
        
        values = np.array(value_list)
        
        return values
    
    
    def iterate(self, fun, minimum_measurements=1, discard_time=False, return_as_map=False, map_default_value=float('nan')):
        if discard_time:
            values = self.iterate_space_discard_time(fun, minimum_measurements=minimum_measurements)
        else:
            values = self.iterate_space_time(fun, minimum_measurements=minimum_measurements)
        
        if return_as_map:
            values = ndop.measurements.util.insert_values_in_map(values, default_value=map_default_value)
        
        return values
    
    
    def number_of_measurements(self, minimum_measurements=1, discard_time=False, return_as_map=False):
        return self.iterate(len, minimum_measurements, discard_time, return_as_map)
    
    
    def means(self, minimum_measurements=1, discard_time=False, return_as_map=False):
        return self.iterate(np.average, minimum_measurements, discard_time, return_as_map)
    
    
    
    def variances(self, minimum_measurements=3, discard_time=False, return_as_map=False):
        def calculate_variance(results):
            mean = np.average(results)
            number_of_results = results.size
            variance = np.sum((results - mean)**2) / (number_of_results - 1)
            return variance
        
        return self.iterate(calculate_variance, minimum_measurements, discard_time, return_as_map)
    
    
    def standard_deviations(self, minimum_measurements=3, discard_time=False, return_as_map=False):
        def calculate_standard_deviation(results):
            mean = np.average(results)
            number_of_results = results.size
            standard_deviation = (np.sum((results - mean)**2) / (number_of_results - 1))**(1/2)
            return standard_deviation
        
        return self.iterate(calculate_standard_deviation, minimum_measurements, discard_time, return_as_map)
    
    
    
#     def variogram_increment_dict(self, space_offset=(0, 0, 0), time_offset=0, minimum_measurements=50):
#         from .constants import SAME_DATETIME_BOUND
#         from ndop.metos3d.constants import METOS_DIM
#         
#         self.print_debug_inc(('Calculating variogram increment dict for space offset ', space_offset, ' and time offset ', time_offset, '.'))
#         
#         transform_time_function = lambda t: math.floor(t / SAME_DATETIME_BOUND) * SAME_DATETIME_BOUND
#         measurements_dict = self.measurements_dict
#         variogram_dict = dict()
#         
#         space_offset_array = np.array(space_offset)
#         
#         ## calculate max space factor
#         if np.all(space_offset_array == 0):
#             max_space_factor = 0
#         else:
#             max_space_factor_dim = np.array(METOS_DIM)
#             max_space_factor_dim[0:1] = max_space_factor_dim[0:1] / 2
#             max_space_factor_mask = space_offset_array != 0
#             max_space_factor = math.ceil(max(max_space_factor_dim[max_space_factor_mask] / space_offset_array[max_space_factor_mask]))
#         
#         ## compute variogram_dict
#         self.print_debug(('Max space factor is ', max_space_factor, '.'))
#         
#         ## iterate over all measurements
#         for (space_index, time_dict) in measurements_dict.items():
#             self.print_debug_inc(('Looking at space index ', space_index, '.'))
#             space_index_array = np.array(space_index)
#             
#             ## iterate over all possible space factors
#             for space_factor in range(max_space_factor + 1):
#                 self.print_debug_inc(('Using space factor ', space_factor, '.'))
#                 
#                 ## calculate space offset
#                 space_offset_index_array = space_index_array + space_factor * space_offset_array
#                 if space_offset_index_array[0] < 0:
#                     space_offset_index_array[0] += METOS_X_DIM
#                 if space_offset_index_array[1] < 0:
#                     space_offset_index_array[1] += METOS_Y_DIM
#                 space_offset_index = tuple(space_offset_index_array)
#             
#                 ## get time dict for space offset 
#                 try:
#                     time_offset_dict = measurements_dict[space_offset_index]
#                 except KeyError:
#                     time_offset_dict = None
#                 
#                 ## iterate over all time combinations
#                 if time_offset_dict is not None:
#                     variogram_time_dict = variogram_dict.setdefault(space_factor, dict())
#                     
#                     for (time_index, results_list) in time_dict.items():
#                         for (time_offset_index, results_offset_list) in time_offset_dict.items():
#                             
#                             if time_offset == 0:
#                                 time_factor = int((time_index - time_offset_index) / SAME_DATETIME_BOUND)
#                             else:
#                                 time_factor = round((time_index - time_offset_index) / time_offset)
#                             
#                             ## insert results in varigram dict
#                             if (time_offset != 0 and time_factor >= 0) or (time_offset == 0 and time_factor == 0):
#                                 
#                                 variogram_increment_list = variogram_time_dict.setdefault(time_factor, [])
#                                 
#                                 for result in results_list:
#                                     for result_offset in results_offset_list:
#                                         increment= (result - result_offset)**2
#                                         variogram_increment_list.append(increment)
#                     
#                     self.print_debug((len(variogram_time_dict.keys()), ' different time factors found.'))
#                 else:
#                     self.print_debug('No time factors found.')
#                 
#                 self.required_debug_level_dec()
#             
#             self.required_debug_level_dec()
#             
#         self.print_debug_idec('Variogram increment dict calculated.')
#         
#         return variogram_increment_dict
#     
#     
#     def variogram(self, space_offset=(0, 0, 0), time_offset=0, minimum_measurements=50):
#         from .constants import SAME_DATETIME_BOUND
#         from ndop.metos3d.constants import METOS_X_DIM, METOS_Y_DIM
#         
#         self.print_debug_inc('Calculating variogram.')
#         
#         variogram_increment_dict = self.variogram_increment_dict(space_offset, time_offset, minimum_measurements)
#         
#         ## compute variogram list
#         self.print_debug('Calculating variogram list.')
#         
#         variogram_list = []
#         for (space_factor, variogram_time_dict) in variogram_dict.items():
#             for (time_factor, variogram_result) in variogram_time_dict.items():
#                 variogram_increment_list = variogram_result
#                 variogram_number = len(variogram_increment_list)
#                 
#                 if variogram_number >= minimum_measurements:
#                     variogram_increment_array = np.array(variogram_increment_list)
#                     variogram_mean = np.average(variogram_increment_array)
#                     variogram_percentiles = np.percentile(variogram_increment_array, (2.5, 50, 97.5), overwrite_input=True)
#                     
#                     variogram_result = (space_factor, time_factor, variogram_mean, variogram_number) + tuple(variogram_percentiles)
#                     variogram_list.append(variogram_result)
#         
#         
#         self.print_debug('Calculating variogram array.')
#         variogram = np.array(variogram_list)
#         
#         self.print_debug_dec('Variogram calculated.')
#         
#         return variogram
    

    
    def iterate_over_shift(self, calculate_function, direction=(0, 0, 0, 0), same_space_bound=[0.5, 0.5, 10], same_time_bound=1/(366.0*2)):
        self.print_debug_inc(('Calculating results list from shifts for direction ', direction, '.'))
        
        measurements_dict = self.measurements_dict
        function_results_list = []
        
        ## prepare directions
        direction_array = np.array(direction)
        space_direction_array = direction_array[0:-1]
        time_direction = direction[-1]
        
        ## calculate max factor
        space_keys = np.array(list(measurements_dict.keys()))
        space_min = np.min(space_keys, axis=0)
        space_max = np.max(space_keys, axis=0)
        
        if np.all(direction == 0):
            max_factor = 0
        else:
            max_factor_dim = np.array(tuple(space_max - space_min) + (100,0))
            max_factor_dim[0] = max_factor_dim[0] / 2
            max_factor_mask = direction_array != 0
            max_factor = math.floor(min(max_factor_dim[max_factor_mask] / direction_array[max_factor_mask]))
        
        self.print_debug(('Max factor is ', max_factor, '.'))
        
        ## calculate x range
        x_range = (space_min[0], space_max[0])
        x_range_diff = x_range[1] - x_range[0]
        def wrap_around_x(x):
            if x < x_range[0]:
                x += x_range_diff
            elif x >= x_range[1]:
                x -= x_range_diff
            return x
        
        self.print_debug(('Range of x is ', x_range, '.'))
        
        
        ## iterate over all factors
        for factor in range(max_factor + 1):
            self.print_debug_inc(('Current factor is ', factor, '.'))
            shift_list = []
                
            ## iterate over all space indices
            for (space_index, time_dict) in measurements_dict.items():
                self.print_debug_inc(('Current space index is ', space_index, '.'))
                
                ## calculate desired space index
                space_index_array = np.array(space_index)
                shifted_space_index_desired_array = space_index_array + factor * space_direction_array
                shifted_space_index_desired_array[0] = wrap_around_x(shifted_space_index_desired_array[0])
                self.print_debug(('Desired space index is ', shifted_space_index_desired_array, '.'))
                
                ## search matching space index
                for (shifted_space_index, shifted_time_dict) in measurements_dict.items():
                    space_diff_array = shifted_space_index_desired_array - space_index_array
                    space_diff_array[0] = wrap_around_x(space_diff_array[0])
                    
                    ## if matching space index found
                    if np.all(np.abs(space_diff_array) <= same_space_bound):
                        self.print_debug(('Matching space index ', shifted_space_index, ' found.'))
                        
                        ## iterate over all time indices
                        for (time_index, results_list) in time_dict.items():
                            shifted_time_index_desired = time_index + factor * time_direction
                            for (shifted_time_index, shifted_results_list) in shifted_time_dict.items():
                                time_diff = shifted_time_index - shifted_time_index_desired
                                
                                ## insert shift to shift list if desired time diff
                                if abs(time_diff) <= same_time_bound:
                                    for result in results_list:
                                        for shifted_results in shifted_results_list:
                                            shift_list.append((result, shifted_results))
                
                self.required_debug_level_dec()
            
            ## apply calculate_function to shift list
            self.print_debug(('Applying calculate function to ', len(shift_list), ' shifts.'))
            
            function_result = calculate_function(shift_list)
            function_results_list.append(function_result)
            
            self.required_debug_level_dec()
            
        self.print_debug_dec('Results list calculated.')
        
        return function_results_list
    
    
    
    def correlogram(self, direction=(0, 0, 0, 0), same_space_bound=[0.5, 0.5, 10], same_time_bound=1/(366.0*2), minimum_measurements=1):
        def calculate_function(shift_list):
            number = len(shift_list)
            
            if number >= minimum_measurements:
                shift_array = np.array(shift_list)
                x = shift_array[:,0]
                y = shift_array[:,1]
                mean_x = np.average(x)
                mean_y = np.average(y)
                sd_x = (np.sum((x - mean_x)**2) / number)**(1/2)
                sd_y = (np.sum((y - mean_y)**2) / number)**(1/2)
                prod_array = ((x - mean_x) * (y - mean_y)) / (sd_x * sd_y)
                correlation = np.sum(prod_array) / number
                percentiles = np.percentile(prod_array, (2.5, 50, 97.5), overwrite_input=True)
                
                result = (correlation,) + tuple(percentiles) + (number,)
            else:
                result = (float('nan'),)*4 + (number,)
            
            return result
        
        
        self.print_debug('Calculating correlation list.')
        
        correlation_list = self.iterate_over_shift(calculate_function, direction, same_space_bound, same_time_bound)
        
        self.print_debug('Calculating correlogram array.')
        correlogram = np.array(correlation_list)
        
        self.print_debug_dec('Correlogram calculated.')
        
        return correlogram
        
    
    
    
#     def calculate_from_increment(self, calculate_function, direction=(0, 0, 0, 0)):
#         from .constants import SAME_DATETIME_BOUND
#         from ndop.metos3d.constants import METOS_DIM
#         
#         self.print_debug_inc(('Calculating results list from increments for direction ', direction, '.'))
#         
#         measurements_dict = self.measurements_dict
#         function_results_list = []
#         
#         ## prepare directions
#         direction_array = np.array(direction)
#         space_direction_array = direction_array[0:-1]
#         time_direction = direction[-1]
#         
#         ## calculate max factor
#         if np.all(direction == 0):
#             max_factor = 0
#         else:
#             max_factor_dim = np.array(METOS_DIM + (100,0))
#             max_factor_dim[0:2] = max_factor_dim[0:2] / 2
#             max_factor_mask = direction_array != 0
#             max_factor = math.floor(min(max_factor_dim[max_factor_mask] / direction_array[max_factor_mask]))
#         
#         self.print_debug(('Max factor is ', max_factor, '.'))
#         
#         ## iterate over all factors
#         for factor in range(max_factor + 1):
#             self.print_debug_inc(('Using factor ', factor, '.'))
#             
#             increment_list = []
#                 
#             ## iterate over all measurements
#             for (space_index, time_dict) in measurements_dict.items():
#                 self.print_debug_inc(('Looking at space index ', space_index, '.'))
#                 space_index_array = np.array(space_index)
#                 
#                 ## calculate space incremented index
#                 space_incremented_index_array = space_index_array + factor * space_direction_array
#                 for i in range(2):
#                     if space_incremented_index_array[i] < 0:
#                         space_incremented_index_array[i] += METOS_DIM[i]
#                     elif space_incremented_index_array[i] >= METOS_DIM[i]:
#                         space_incremented_index_array[i] -= METOS_DIM[i]
#                 space_incremented_index = tuple(space_incremented_index_array)
#                 
#                 self.print_debug(('Space incremented index is ', space_incremented_index, '.'))
#                 
#                 ## get time dict for space incremented index 
#                 try:
#                     time_incremented_dict = measurements_dict[space_incremented_index]
#                 except KeyError:
#                     time_incremented_dict = None
#                 
#                 
#                 ## iterate over all time combinations
#                 if time_incremented_dict is not None:
#                     for (time_index, results_list) in time_dict.items():
#                         time_incremented_index_desired = time_index + factor * time_direction
#                         for (time_incremented_index, results_incremented_list) in time_incremented_dict.items():
#                             time_diff = time_incremented_index - time_incremented_index_desired
#                             
#                             ## insert increment to increment list if desired time diff
#                             if time_diff >= 0 and time_diff <= SAME_DATETIME_BOUND:
#                                 for result in results_list:
#                                     for result_incremented in results_incremented_list:
#                                         increment= (result_incremented - result)**2
#                                         increment_list.append(increment)
#                 
#                 self.required_debug_level_dec()
#             
#             ## apply calculate_function to increment list
#             self.print_debug(('Applying calculate function to ', len(increment_list), ' increments.'))
#             
#             function_result = calculate_function(increment_list)
#             function_results_list.append(function_result)
#             
#             self.required_debug_level_dec()
#             
#         self.print_debug_dec('Results list calculated.')
#         
#         return function_results_list
    
    
    
    def variogram(self, direction=(0, 0, 0, 0)):
        def calculate_function(increment_list):
            number = len(increment_list)
            
            if number > 0:
                increment_array = np.array(increment_list) / 2
                mean = np.average(increment_array)
                percentiles = np.percentile(increment_array, (2.5, 50, 97.5), overwrite_input=True)
                
                result = (mean,) + tuple(percentiles) + (number,)
            else:
                result = (float('nan'),)*4 + (0,)
            
            return result
        
        
        self.print_debug('Calculating variogram list.')
        
        variogram_list = self.calculate_from_increment(calculate_function, direction)
        
        
#         variogram_list = []
#         for (space_factor, variogram_time_dict) in variogram_dict.items():
#             for (time_factor, variogram_result) in variogram_time_dict.items():
#                 variogram_result = (space_factor, time_factor) + variogram_result
#                 variogram_list.append(variogram_result)
        
        self.print_debug('Calculating variogram array.')
        variogram = np.array(variogram_list)
        
        self.print_debug_dec('Variogram calculated.')
        
        return variogram
        
        ##################
        #TODO: plot (averaged) increment per space
        #TODO: discard (different) axis at measurements
        #TODO: transform at all axis possible
        ##################
    
    
    
    
#     def variogram(self, space_offset=(0, 0, 0), time_offset=0, minimum_measurements=50):
#         from .constants import SAME_DATETIME_BOUND
#         from ndop.metos3d.constants import METOS_X_DIM, METOS_Y_DIM
#         
#         self.print_debug_inc('Calculating variogram.')
#         
#         transform_time_function = lambda t: math.floor(t / SAME_DATETIME_BOUND) * SAME_DATETIME_BOUND
#         measurements_dict = self.transform_time(transform_time_function).measurements_dict
#         variogram_dict = dict()
#         
#         space_offset_array = np.array(space_offset)
#         if np.all(space_offset_array == 0):
#             max_space_factor = 1
#         else:
#             max_space_factor = METOS_X_DIM
#         
#         ## compute variogram_dict
#         self.print_debug('Calculating variogram dict.')
#         
#         ## iterate over all measurements
#         for (space_index, time_dict) in measurements_dict.items():
#             self.print_debug_inc(('Looking at space index ', space_index, '.'))
#             space_index_array = np.array(space_index)
#             
#             ## iterate over all possible space factors
#             for space_factor in range(max_space_factor):
#                 self.print_debug_inc(('Using space factor ', space_factor, '.'))
#                 
#                 ## calculate space offset
#                 space_offset_index_array = space_index_array + space_factor * space_offset_array
#                 if space_offset_index_array[0] < 0:
#                     space_offset_index_array[0] += METOS_X_DIM
#                 if space_offset_index_array[1] < 0:
#                     space_offset_index_array[1] += METOS_Y_DIM
#                 space_offset_index = tuple(space_offset_index_array)
#             
#                 ## get time dict for space offset 
#                 try:
#                     time_offset_dict = measurements_dict[space_offset_index]
#                 except KeyError:
#                     time_offset_dict = None
#                 
#                 ## iterate over all time combinations
#                 if time_offset_dict is not None:
#                     variogram_time_dict = variogram_dict.setdefault(space_factor, dict())
#                     
#                     for (time_index, results_list) in time_dict.items():
#                         for (time_offset_index, results_offset_list) in time_offset_dict.items():
#                             
#                             if time_offset == 0:
#                                 time_factor = 0
#                             else:
#                                 time_factor = round((time_index - time_offset_index) / time_offset)
#                             
#                             ## insert results in varigram dict
#                             if time_offset != 0 or time_index == time_offset_index:
#                                 
#                                 (variogram_sum, variogram_number) = variogram_time_dict.setdefault(time_factor, (0, 0))
#                                 
#                                 for result in results_list:
#                                     for result_offset in results_offset_list:
#                                         variogram_sum += (result - result_offset)**2
#                                 variogram_number += len(results_list) * len(results_offset_list)
#                                 
#                                 variogram_time_dict[time_factor] = (variogram_sum, variogram_number)
#                     
#                     
#                     self.print_debug_inc_dec((variogram_number, ' measurement combinations used.'))
#                 
#                 self.required_debug_level_dec()
#             
#             self.required_debug_level_dec()
#         
#         
#         ## compute variogram list
#         self.print_debug('Calculating variogram list.')
#         
#         variogram_list = []
#         for (space_factor, variogram_time_dict) in variogram_dict.items():
#             for (time_factor, variogram_result) in variogram_time_dict.items():
#                 (variogram_sum, variogram_number) = variogram_result
#                 
#                 if variogram_number >= minimum_measurements:
#                     variogram_result = variogram_sum / (2 * variogram_number) 
#                     variogram_list.append((space_factor, time_factor, variogram_result))
#         
#         
#         self.print_debug('Calculating variogram array.')
#         variogram = np.array(variogram_list)
#         
#         self.print_debug_dec('Variogram calculated.')
#         
#         return variogram
#         
#         
#             
#             #####################
#             for (time_index, results_list) in time_dict.items():
#                 
#                 space_offset_index = (space_index[0] + space_offset[0], space_index[1] + space_offset[1], space_index[2] + space_offset[2])
#                 time_offset_index = time_index + time_offset
#                 
#                 
#                 
#                 ## get results for time offset
#                 try:
#                     results_offset_list = time_offset_dict[time_offset_index]
#                 except KeyError:
#                     results_offset_list = []
#                 
#                 
# #                 for (time_index_spatial_offset, results_offset_list) in time_offset_dict.items():
# #                     if abs(time_index - time_index_spatial_offset) <= SAME_DATETIME_BOUND:
#                         
#                 
#                 
#                 
#                 if len(results_list) >= minimum_measurements:
#                     index = space_index + (time_index,)
#                     results = np.array(results_list)
#         
#         
#         measurements_dict = self.measurements_dict
#         variogram_measurements_dict = dict()
#         
#         for (space_index, time_dict) in measurements_dict.items():
#             variogram_time_dict = variogram_measurements_dict.setdefault(space_index, dict())
#             
#             for (time_index, results_list) in time_dict.items():
#                 variogram_time_index = transform_function(time_index)
#                 variogram_result_list = variogram_time_dict.setdefault(variogram_time_index, [])
#                 variogram_result_list += results_list
#         
#         variogram_measurements = Measurements(debug_level=self.debug_level, required_debug_level=self.required_debug_level)
#         variogram_measurements.measurements_dict = variogram_measurements_dict
#         
#         return variogram_measurements
        
    
    
#     def get_correlation(self, time_offset=0, spatial_offset=(0,0,0), minimum_measurements=3):
#         from .constants import SAME_DATETIME_BOUND
#         measurements_dict = self.measurements_dict
#         spatial_indices_list = []
#         value_list = []
#         
#         for (space_index, time_results_list) in measurements_dict.items:
#             if len(time_results_list) >= minimum_measurements:
#                 space_index_offset = tuple(map(sum, zip(space_index, spatial_offset)))
#                 time_results_list_offset = measurements_dict[space_index_offset]
#                 if len(time_results_list_offset) >= minimum_measurements:
#                     matching_results_list = []
#                     
#                     ## find matching results
#                     for i in range(len(time_results_list)):
#                         time_i, result_i = time_results_list[i]
#                         for j in range(len(time_results_list_offset)):
#                             time_j, result_j = time_results_list_offset[j]
#                             
#                             if abs(time_i - time_j - time_offset) < SAME_DATETIME_BOUND
#                                 matching_results_list.append((result_i, result_j))
#                     
#                     ## if enough matching results, calculate covariance
#                     if len(matching_results_list) >= minimum_measurements:
#                         spatial_indices_list.append(space_index)
#                 
#                 
#                 spatial_indices_list.append(space_index)
#                 
#                 results = np.array(results)[1].astype(float)
#                 value = fun(results)
#                 
#                 value_list.append(value)
#         
#         spatial_indices = np.array(spatial_indices_list, dtype=np.uint16)
#         values = np.array(value_list)
#         
#         return (spatial_indices, values)
        
        
        
# #         measurement_results_dict = self.measurement_results
# #         measurements_list = []
# #         value_list = []
# #         
# #         for (measurement, results) in measurement_results_dict.items:
# #             if len(results) >= minimum_measurements:
# #                 tuple(map(sum, zip((1, 2), (3, 4))))
# #                 
# #                 measurements_list.append(measurement)
# #                 
# #                 results = np.array(results)[1].astype(float)
# #                 value = fun(results)
# #                 
# #                 value_list.append(value)
# #                 
# #                 tuple(map(sum, zip((1, 2), (3, 4))))
# #         
# #         measurements = np.array(measurements_list, dtype=np.uint16)
# #         values = np.array(value_list)
# #     
# # #     def get_means(self, minimum_measurements=1):
# # #         measurement_results_dict = self.measurement_results
# # #         measurements_list = []
# # #         means_list = []
# # #         
# # #         for (measurement, results) in measurement_results_dict.values:
# # #             if len(results) >= minimum_measurements:
# # #                 measurements_list.append(measurement)
# # #                 
# # #                 results = np.array(results)
# # #                 mean = np.average(results)
# # #                 
# # #                 means_list.append(mean)
# # #         
# # #         measurements = np.array(measurements_list)
# # #         means = np.array(means_list)
# # #         
# # #         return (measurements, means)
# # #     
# # #     
# # #     def get_standard_deviations(self, minimum_measurements=3):
# # #         measurement_results_dict = self.measurement_results
# # #         measurements_list = []
# # #         standard_deviations_list = []
# # #         
# # #         for (measurement, results) in measurement_results_dict.values:
# # #             if len(results) >= minimum_measurements:
# # #                 measurements_list.append(measurement)
# # #                 
# # #                 results = np.array(results)
# # #                 mean = np.average(results)
# # #                 number_of_results = results.size
# # #                 standard_deviation = (np.sum((results - mean)**2) / (number_of_results - 1))**(1/2)
# # #                 
# # #                 standard_deviations_list.append(standard_deviation)
# # #         
# # #         measurements = np.array(measurements_list)
# # #         standard_deviations = np.array(standard_deviations_list)
# # #         
# # #         return (measurements, standard_deviations)
# #     
# #     
# #     def get_correlation(self, minimum_measurements=3):
# #         
# #         measurements, means = self.get_means(minimum_measurements=minimum_measurements)
# #         
# #         measurement_results_dict = self.measurement_results
# #         number_of_measurements = means.size
# #         
# #         correlation = np.empty((number_of_measurements, number_of_measurements)) * np.nan
# #         
# #         for i in range(number_of_measurements):
# #             mean_i = means[i]
# #             for j in range(i+1, number_of_measurements):
# #                 mean_j = means[j]
# #                 
# #             
# #         
# #         
# #         
# #         measurements_list = []
# #         means_list = []
# #         standard_deviations_list = []
# #         
# #         for (measurement, results) in measurement_results_dict.values:
# #             if len(results) >= minimum_measurements:
# #                 measurements_list.append(measurement)
# #                 
# #                 results = np.array(results)
# #                 mean = np.average(results)
# #                 means_list.append(mean)
# #                 standard_deviation = np.average((results - mean)**2)
# #                 
# #                 standard_deviations_list.append(standard_deviation)
# #         
# #         measurements = np.array(measurements_list)
# #         standard_deviations = np.array(standard_deviations_list)
# #         
# #         return (measurements, standard_deviations)
    
#         
#     
#     
#     def get_standard_deviation(self):
#         
#     
#     def get_covariance(self, cruise_collection, minimum_measurements=3):
#         