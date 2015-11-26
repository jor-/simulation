import bisect
import numpy as np

import ndop.util.value_cache
import ndop.model.eval

import measurements.all.box.data
import measurements.all.pw.values
import measurements.all.pw.correlation
import measurements.all.pw_nearest.data
import measurements.all.pw_nearest.correlation
import measurements.land_sea_mask.data
import measurements.util.data

import util.math.matrix
import util.cache

import util.logging
logger = util.logging.logger




class DataBase:

    def __init__(self, spinup_options=None, derivative_options=None, time_step=1, parameter_tolerance_options=None, job_setup=None):
        from .constants import CACHE_DIRNAME, BOXES_F_FILENAME, BOXES_DF_FILENAME

        logger.debug('Initiating {} with spinup_options {}, derivative_options {}, time step {}, parameter_tolerance_options {} and job_setup {}.'.format(self, spinup_options, derivative_options, time_step, parameter_tolerance_options, job_setup))
        
        self.model = ndop.model.eval.Model(job_setup=job_setup, spinup_options=spinup_options, derivative_options=derivative_options, time_step=time_step, parameter_tolerance_options=parameter_tolerance_options)

        self.f_boxes_cache_filename = BOXES_F_FILENAME
        self.df_boxes_cache_filename = BOXES_DF_FILENAME
        
        self.hdd_cache = ndop.util.value_cache.Cache(spinup_options=self.model.spinup_options, derivative_options=self.model.derivative_options, time_step=self.model.time_step, parameter_tolerance_options=parameter_tolerance_options, cache_dirname=CACHE_DIRNAME, use_memory_cache=True)
        self.memory_cache = util.cache.MemoryCache()
        self.memory_cache_with_parameters = ndop.util.value_cache.MemoryCache()

        if job_setup is None:
            job_setup = {}
        try:
            job_setup['name']
        except KeyError:
            job_setup['name'] = str(self)


    def __str__(self):
        return self.__class__.__name__



    ## model output
    
    def f_boxes_calculate(self, parameters, time_dim=12):
        logger.debug('Calculating new model f_boxes with time dim {} for {}.'.format(time_dim, self))
        f_boxes = self.model.f_boxes(parameters, time_dim_desired=time_dim)
        f_boxes = np.asanyarray(f_boxes)
        return f_boxes

    def f_boxes(self, parameters, time_dim=12, use_memmap=False):
        calculation_function = lambda p: self.f_boxes_calculate(p, time_dim=time_dim)
        return self.hdd_cache.get_value(parameters, self.f_boxes_cache_filename.format(time_dim), calculation_function, derivative_used=False, save_also_txt=False, use_memmap=use_memmap)



    def df_boxes_calculate(self, parameters, time_dim=12):
        logger.debug('Calculating new model df_boxes with time dim {} for {}.'.format(time_dim, self))
        df_boxes = self.model.df_boxes(parameters, time_dim_desired=time_dim)
        df_boxes = np.asanyarray(df_boxes)
        for i in range(1, df_boxes.ndim-1):
            df_boxes = np.swapaxes(df_boxes, i, i+1)
        return df_boxes

    def df_boxes(self, parameters, time_dim=12, use_memmap=False, as_shared_array=False):
        calculation_function = lambda p: self.df_boxes_calculate(p, time_dim=time_dim)
        return self.hdd_cache.get_value(parameters, self.df_boxes_cache_filename.format(time_dim), calculation_function, derivative_used=True, save_also_txt=False, use_memmap=use_memmap, as_shared_array=as_shared_array)



    def F_calculate(self, parameters):
        raise NotImplementedError("Please implement this method.")
    
    def F(self, parameters):
        values = self.memory_cache_with_parameters.get_value(parameters, 'F', self.F_calculate)
        assert values.ndim == 1 and len(values) == self.m
        return values

    def DF_calculate(self, parameters):
        raise NotImplementedError("Please implement this method.")

    def DF(self, parameters):
        values = self.memory_cache_with_parameters.get_value(parameters, 'DF', self.DF_calculate)
        assert values.ndim == 2 and len(values) == self.m and values.shape[1] == len(parameters)
        return values


    ## m
    
    @property
    def m(self):
        return self.memory_cache.get_value('m', lambda: len(self.results))

    ## results

    def results_calculate(self):
        raise NotImplementedError("Please implement this method")

    @property
    def results(self):
        values = self.memory_cache.get_value('results', lambda: self.results_calculate())
        assert values.ndim == 1
        return values


    ## deviation

    def deviations_calculate(self):
        raise NotImplementedError("Please implement this method")

    @property
    def deviations(self):
        values = self.memory_cache.get_value('deviations', lambda: self.deviations_calculate())
        assert values.ndim == 1 and len(values) == self.m
        return values

    @property
    def inverse_deviations(self):
        values = self.memory_cache.get_value('inverse_deviations', lambda: 1 / self.deviations)
        assert values.ndim == 1 and len(values) == self.m
        return values

    @property
    def variances(self):
        values = self.memory_cache.get_value('variances', lambda: self.deviations**2)
        assert values.ndim == 1 and len(values) == self.m
        return values

    @property
    def inverse_variances(self):
        values = self.memory_cache.get_value('inverse_variances', lambda: 1 / self.variances)
        assert values.ndim == 1 and len(values) == self.m
        return values

    @property
    def average_variance(self):
        values = self.memory_cache.get_value('average_variance', lambda: self.variances.mean())
        assert np.isfinite(values)
        return values

    @property
    def inverse_average_variance(self):
        values = self.memory_cache.get_value('inverse_average_variance', lambda: 1 / self.average_variance)
        assert np.isfinite(values)
        return values



    ## deviation boxes

    def deviations_boxes(self, time_dim=12, as_shared_array=False):
        return self.memory_cache.get_value('deviations_boxes_{}'.format(time_dim), lambda: measurements.all.pw.values.deviation_TMM(t_dim=time_dim), as_shared_array=as_shared_array)

    def inverse_deviations_boxes(self, time_dim=12, as_shared_array=False):
        return self.memory_cache.get_value('inverse_deviations_boxes_{}'.format(time_dim), lambda: 1 / self.deviations_boxes(time_dim=time_dim), as_shared_array=as_shared_array)



class DataBaseHDD(DataBase):
    
    def __init__(self, *args, F_cache_filename=None, DF_cache_filename=None, **kargs):
        logger.debug('Initiating {} with F_cache_filename {} and DF_cache_filename {}.'.format(self, F_cache_filename, DF_cache_filename))
        super().__init__(*args, **kargs)
        self._F_cache_filename = F_cache_filename
        self._DF_cache_filename = DF_cache_filename

    @property
    def F_cache_filename(self):
        return self._F_cache_filename

    @property
    def DF_cache_filename(self):
        return self._DF_cache_filename.format(step_size=self.model.derivative_options['step_size'])
        
    
    def F(self, parameters):
        values = self.hdd_cache.get_value(parameters, self.F_cache_filename, self.F_calculate, derivative_used=False, save_also_txt=False)
        assert values.ndim == 1 and len(values) == self.m
        return values

    def DF(self, parameters):
        values = self.hdd_cache.get_value(parameters, self.DF_cache_filename, self.DF_calculate, derivative_used=True, save_also_txt=False)
        assert values.ndim == 2 and len(values) == self.m and values.shape[1] == len(parameters)
        return values
    



class WOA(DataBaseHDD):

    def __init__(self, spinup_options=None, derivative_options=None, time_step=1, parameter_tolerance_options=None, job_setup=None):
        ## super constructor
        from .constants import WOA_F_FILENAME, WOA_DF_FILENAME
        super().__init__(sspinup_options=spinup_options, derivative_options=derivative_options, time_step=time_step, parameter_tolerance_options=parameter_tolerance_options, job_setup=job_setup, F_cache_filename=WOA_F_FILENAME, DF_cache_filename=WOA_DF_FILENAME)

        ## compute annual box index
        from measurements.po4.woa.data13.constants import ANNUAL_THRESHOLD
        from ndop.model.constants import METOS_Z_LEFT
        self.ANNUAL_THRESHOLD_INDEX = bisect.bisect_right(METOS_Z_LEFT, ANNUAL_THRESHOLD)



    def _get_data_with_annual_averaged(self, data, annual_factor=1):
        data_monthly = data[:,:,:,:,:self.ANNUAL_THRESHOLD_INDEX][self.mask[:,:,:,:,:self.ANNUAL_THRESHOLD_INDEX]]
        data_annual = np.average(data[:,:,:,:,self.ANNUAL_THRESHOLD_INDEX:], axis=1)[self.mask[:,0,:,:,self.ANNUAL_THRESHOLD_INDEX:]] * annual_factor
        return np.concatenate([data_monthly, data_annual], axis=0)


    ## model output

    def F_calculate(self, parameters):
        f_boxes = self.f_boxes(parameters)
        F = self._get_data_with_annual_averaged(f_boxes)
        return F


    def DF_calculate(self, parameters):
        df_boxes = self.df_boxes(parameters)
        DF = self._get_data_with_annual_averaged(df_boxes)
        return DF


    ## devitation

    @property
    def mean_deviations_boxes(self):
        return self.memory_cache.get_value('mean_deviations_boxes', lambda: (measurements.all.box.data.variances() / measurements.all.box.data.nobs())**(1/2))

    def deviations_calculate(self):
        return self._get_data_with_annual_averaged(self.mean_deviations_boxes, annual_factor=1/12)


    ## measurements

    @property
    def mask(self):
        return self.memory_cache.get_value('mask', lambda: measurements.all.box.data.nobs() > 0)

    @property
    def results_boxes(self):
        return self.memory_cache.get_value('results_boxes', lambda: measurements.all.box.data.means())

    def results_calculate(self):
        return self._get_data_with_annual_averaged(self.results_boxes)


    ## diff

    def diff_boxes(self, parameters, normalize_with_deviation=False, no_data_value=np.inf):
        results_boxes = self.results_boxes
        results_boxes[np.logical_not(self.mask)] = no_data_value
        diff = results_boxes - self.f_boxes(parameters)
        if normalize_with_deviation:
            diff = diff / self.mean_deviations_boxes
        return diff





class WOD_Base(DataBase):

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)


    ## model output

    def F_dop(self, parameters):
        F = self.F(parameters)
        return F[:self.m_dop]


    def F_po4(self, parameters):
        F = self.F(parameters)
        return F[self.m_dop:]


    ## measurements

    @property
    def points(self):
        values = self.memory_cache.get_value('points', lambda: self.points_calculate())
        assert len(values) == 2 
        assert all([ndim == 2 for ndim in map(lambda a: a.ndim, values)])
        return values


    @property
    def m_dop(self):
        return self.memory_cache.get_value('m_dop', lambda: len(self.points[0]))

    @property
    def m_po4(self):
        return self.memory_cache.get_value('m_po4', lambda: len(self.points[1]))

    @property
    def m(self):
        m = super().m
        assert self.m_dop + self.m_po4
        return m


    ## correlation matrix

    def correlation_matrix_calculate(self, min_values, max_year_diff=float('inf')):
        raise NotImplementedError("Please implement this method.")

    def correlation_matrix(self, min_values, max_year_diff=float('inf')):
        return self.memory_cache.get_value('correlation_matrix_{:0>2}_{:0>2}'.format(min_values, max_year_diff), lambda: self.correlation_matrix_calculate(min_values, max_year_diff=max_year_diff))


    def correlation_matrix_cholesky_decomposition_calculate(self, min_values, max_year_diff=float('inf')):
        raise NotImplementedError("Please implement this method.")

    def correlation_matrix_cholesky_decomposition(self, min_values, max_year_diff=float('inf')):
        return self.memory_cache.get_value('correlation_matrix_cholesky_decomposition_{:0>2}_{:0>2}'.format(min_values, max_year_diff), lambda: self.correlation_matrix_cholesky_decomposition_calculate(min_values, max_year_diff=max_year_diff))




class WOD(DataBaseHDD, WOD_Base):

    def __init__(self, spinup_options=None, derivative_options=None, time_step=1, parameter_tolerance_options=None, job_setup=None):
        from .constants import WOD_F_FILENAME, WOD_DF_FILENAME
        super().__init__(spinup_options=spinup_options, derivative_options=derivative_options, time_step=time_step, parameter_tolerance_options=parameter_tolerance_options, job_setup=job_setup, F_cache_filename=WOD_F_FILENAME, DF_cache_filename=WOD_DF_FILENAME)


    ## model output

    def F_calculate(self, parameters):
        (f_dop, f_po4) = self.model.f_points(parameters, self.points)
        F = np.concatenate([f_dop, f_po4])
        return F


    def DF_calculate(self, parameters):
        (df_dop, df_po4) = self.model.df_points(parameters, self.points)
        DF = np.concatenate([df_dop, df_po4], axis=-1)
        DF = np.swapaxes(DF, 0, 1)
        return DF



    ## deviation

    def deviations_calculate(self):
        (deviation_dop, deviation_po4) = measurements.all.pw.values.deviation()
        deviations = np.concatenate([deviation_dop, deviation_po4])
        return deviations


    ## measurements

    def points_calculate(self):
        return measurements.all.pw.values.points()


    def results_calculate(self):
        results = np.concatenate(measurements.all.pw.values.results())
        return results


    ## correlation matrix

    def correlation_matrix_calculate(self, min_values, max_year_diff=float('inf')):
        return measurements.all.pw.correlation.CorrelationMatrix(min_values=min_values, max_year_diff=max_year_diff).correlation_matrix_positive_definite


    def correlation_matrix_cholesky_decomposition_calculate(self, min_values, max_year_diff=float('inf')):
        return measurements.all.pw.correlation.CorrelationMatrix(min_values=min_values, max_year_diff=max_year_diff).correlation_matrix_cholesky_decomposition



    ## correlation methods for P3 correlation

    def correlation_parameters(self, parameters):
        from ndop.optimization.constants import COST_FUNCTION_CORRELATION_PARAMETER_FILENAME
        correlation_parameters = self.get_file(parameters, COST_FUNCTION_CORRELATION_PARAMETER_FILENAME)
        return correlation_parameters


    def check_regularity(self, correlation_parameters):
        [a, b, c] = correlation_parameters
        n = self.m_dop
        m = self.m_po4

        regularity_factor = (1+(n-1)*a) * (1+(m-1)*b) - n*m*c**2

        if regularity_factor <= 0:
            raise util.math.matrix.SingularMatrixError('The correlation matrix with correlation parameters (a, b, c) = {} is singular. It has to be (1+(n-1)*a) * (1+(m-1)*b) - n*m*c**2 > 0'.format(correlation_parameters))


    def ln_det_correlation_matrix(self, correlation_parameters):
        [a, b, c] = correlation_parameters
        n = self.m_dop
        m = self.m_po4

        self.check_regularity(correlation_parameters)
        ln_det = (n-1)*np.log(1-a) + (m-1)*np.log(1-b) + np.log((1+(n-1)*a) * (1+(m-1)*b) - n*m*c**2)

        return ln_det


    def project(self, values, split_index, projected_value_index=None):
#         if len(values) != 2:
#             raise ValueError('Values must be a list with length 2, but its length is {}.'.format(len(values)))

        if projected_value_index in (0, 1):
            values = (values[:split_index], values[split_index:])

            if projected_value_index == 0:
                values_squared = []

                for i in range(len(values)):
                    value = values[i]
                    value_matrix = util.math.matrix.convert_to_matrix(value, dtype=np.float128)
                    value_squared = np.array(value_matrix.T * value_matrix)
                    value_squared = util.math.matrix.convert_matrix_to_array(value_squared, dtype=np.float128)
                    values_squared.append(value_squared)

                values_squared = np.array(values_squared)

                return values_squared

            elif projected_value_index == 1:
                values_summed = []

                for i in range(len(values)):
                    value = values[i]
                    value_summed = np.sum(value, axis=0, dtype=np.float128)

                    values_summed.append(value_summed)

                values_summed = np.array(values_summed)

                return values_summed

        elif projected_value_index is None:
            return (self.project(values, split_index, projected_value_index=0), self.project(values, split_index, projected_value_index=1))

        else:
            raise ValueError('Unknown projected_value_index: projected_value_index must be 0, 1 or None.')



    def projected_product_inverse_correlation_matrix_both_sides(self, factor_projected, correlation_parameters):
        ## unpack
        assert len(factor_projected) == 2
        (factor_squared, factor_summed) = factor_projected
        assert len(factor_squared) == 2
        assert len(factor_summed) == 2

        ## get correlation parameters
        [a, b, c] = correlation_parameters
        n = self.m_dop
        m = self.m_po4

        ## check regularity
        self.check_regularity(correlation_parameters)

        ## calculate first product part
        product = factor_squared[0] / (1-a) + factor_squared[1] / (1-b)

        ## calculate core matrix (2x2)
        A = np.matrix([[a, c], [c, b]])
        W = np.matrix([[1-a, 0], [0, 1-b]])
        D = np.matrix([[n, 0], [0, m]])
        H = W + D * A
        core = W.I * A * H.I * W.I

        ## calculate second product part
        factor_summed_matrix = util.math.matrix.convert_to_matrix(factor_summed)
        product += factor_summed_matrix.T * core * factor_summed_matrix

        ## return product
        product = np.array(product)
        if product.size == 1:
            product = product[0, 0]

        return product



    ## diff

    def diff(self, parameters, normalize_with_deviation=False):
        diff = self.results - self.F(parameters)
        if normalize_with_deviation:
            diff = diff / self.deviations
        return diff


    def convert_to_boxes(self, data, t_dim=12, no_data_value=np.inf):
        def convert_to_boxes_with_points(points, data):
            assert len(points) == len(data)

            lsm = measurements.land_sea_mask.data.LandSeaMaskTMM(t_dim=t_dim, t_centered=False)
            m = measurements.util.data.Measurements()
            m.append_values(points, data)
            m.transform_indices_to_lsm(lsm)
            data_map = lsm.insert_index_values_in_map(m.means(), no_data_value=no_data_value)

            return data_map

        data_dop_map = convert_to_boxes_with_points(self.points[0], data[:self.m_dop])
        data_po4_map = convert_to_boxes_with_points(self.points[1], data[self.m_dop:])
        data_map = [data_dop_map, data_po4_map]

        return data_map



class WOD_TMM(WOD_Base):
    
    def __init__(self, *args, max_land_boxes=0, **kargs):
        self.max_land_boxes = max_land_boxes
        logger.debug('Initiating {} with max_land_boxes {}.'.format(self, max_land_boxes))
        super().__init__(*args, **kargs)
        self.wod = WOD(*args, **kargs)
        self.lsm = measurements.land_sea_mask.data.LandSeaMaskTMM()
    

    def __str__(self):
        return '{}_{}'.format(self.__class__.__name__, self.max_land_boxes)


    def points_near_water_mask_calculate(self):
        return measurements.all.pw_nearest.data.points_near_water_mask(self.lsm, max_land_boxes=self.max_land_boxes)

    @property
    def points_near_water_mask(self):
        return self.memory_cache.get_value('points_near_water_mask', lambda: self.points_near_water_mask_calculate())

    @property
    def points_near_water_mask_concatenated(self):
        return np.concatenate(self.points_near_water_mask)
    

    def points_calculate(self):
        points_near_water_mask = self.points_near_water_mask
        points = self.wod.points
        points = list(points)
        for i in range(len(points)):
            points[i] = points[i][points_near_water_mask[i]]
        return points


    def results_calculate(self):
        return self.wod.results[self.points_near_water_mask_concatenated]


    def deviations_calculate(self):
        return self.wod.deviations[self.points_near_water_mask_concatenated]


    def F_calculate(self, parameters):
        return self.wod.F(parameters)[self.points_near_water_mask_concatenated]

    def DF_calculate(self, parameters):
        return self.wod.DF(parameters)[self.points_near_water_mask_concatenated]


    ## correlation matrix

    def correlation_matrix_calculate(self, min_values, max_year_diff=float('inf')):
        return measurements.all.pw_nearest.correlation.CorrelationMatrix(min_values=min_values, max_year_diff=max_year_diff, lsm=self.lsm, max_land_boxes=self.max_land_boxes).correlation_matrix_positive_definite

    def correlation_matrix_cholesky_decomposition_calculate(self, min_values, max_year_diff=float('inf')):
        return measurements.all.pw_nearest.correlation.CorrelationMatrix(min_values=min_values, max_year_diff=max_year_diff, lsm=self.lsm, max_land_boxes=self.max_land_boxes).correlation_matrix_cholesky_decomposition

    


## init

def init_data_base(data_kind, spinup_options=None, derivative_options=None, time_step=1, parameter_tolerance_options=None, job_setup=None):
    db_args = ()
    db_kargs = {'spinup_options': spinup_options, 'derivative_options': derivative_options, 'time_step':time_step, 'parameter_tolerance_options': parameter_tolerance_options, 'job_setup': job_setup}
    if data_kind.upper() == 'WOA':
        return WOA(*db_args, **db_kargs)
    elif data_kind.upper() == 'WOD':
        return WOD(*db_args, **db_kargs)
    elif data_kind.upper().startswith('WOD'):
        data_kind_splitted = data_kind.split('.')
        assert len(data_kind_splitted) == 2 and data_kind_splitted[0] == 'WOD'
        max_land_boxes = int(data_kind_splitted[1])
        return WOD_TMM(*db_args, max_land_boxes=max_land_boxes, **db_kargs)
    else:
        raise ValueError('Data_kind {} unknown. Must be "WOA", "WOD" or "WOD.".'.format(data_kind))



## Family

class Family():
    
    member_classes = {}
    
    def __init__(self, **cf_kargs):
        ## chose member classes
        data_kind = cf_kargs['data_kind'].upper()
        try:
            member_classes_list = self.member_classes[data_kind]
        except KeyError:
            raise ValueError('Data_kind {} unknown. Must be in {}.'.format(data_kind, list(self.member_classes.keys())))

        ## init members
        family = []
        for member_class, additional_arguments in member_classes_list:
            for additional_kargs in additional_arguments:
                cf_kargs_member_class = cf_kargs.copy()
                cf_kargs_member_class.update(additional_kargs)
                member = member_class(**cf_kargs_member_class)
                family.append(member)
        
        ## set same database
        for i in range(1, len(family)):
            family[i].data_base = family[0].data_base

        ## set family
        logger.debug('Cost function family for data kind {} with members {} initiated.'.format(data_kind, list(map(lambda x: str(x), family))))
        self.family = family


    def get_function_value(self, function):
        assert callable(function)

        value = function(self.family[0])
        for member in self.family:
            function(member)

        return value