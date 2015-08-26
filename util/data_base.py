import bisect
import numpy as np

import ndop.util.value_cache
from ndop.model.eval import Model

import measurements.all.box.data
import measurements.all.pw.values
import measurements.all.pw.correlation
import measurements.land_sea_mask.data
import measurements.util.data

import util.math.matrix
import util.cache

import util.logging
logger = util.logging.logger


DEFAULT_SPINUP_OPTIONS={'years':10000, 'tolerance':0.0, 'combination':'or'}


class DataBase:

    def __init__(self, spinup_options=DEFAULT_SPINUP_OPTIONS, time_step=1, df_accuracy_order=2, job_setup=None, F_cache_filename=None, DF_cache_filename=None):
        from .constants import CACHE_DIRNAME, F_BOXES_CACHE_FILENAME, DF_BOXES_CACHE_FILENAME

        logger.debug('Initiating {} with spinup_options {}, time step {}, df_accuracy_order {}, job_setup {}, F_cache_filename {} and DF_cache_filename {}.'.format(self, spinup_options, time_step, df_accuracy_order, job_setup, F_cache_filename, DF_cache_filename))

        self.spinup_options = spinup_options
        self.time_step = time_step
        self.df_accuracy_order = df_accuracy_order

        self.cache = ndop.util.value_cache.Cache(spinup_options, time_step, df_accuracy_order=df_accuracy_order, cache_dirname=CACHE_DIRNAME, use_memory_cache=True)
        self.f_boxes_cache_filename = F_BOXES_CACHE_FILENAME
        self.df_boxes_cache_filename = DF_BOXES_CACHE_FILENAME
        self.F_cache_filename = F_cache_filename
        self.DF_cache_filename = DF_cache_filename

        self.memory_cache = util.cache.MemoryCache()

        if job_setup is None:
            job_setup = {}
        try:
            job_setup['name']
        except KeyError:
            job_setup['name'] = str(self)
        self.model = Model(job_setup=job_setup)


    def __str__(self):
        return self.__class__.__name__



    ## model output
    def f_boxes_calculate(self, parameters, time_dim=12):
        logger.debug('Calculating new model f_boxes with time dim {} for {}.'.format(time_dim, self))
        f_boxes = self.model.f_boxes(parameters, time_dim_desired=time_dim, spinup_options=self.spinup_options, time_step=self.time_step)
        f_boxes = np.asanyarray(f_boxes)
        return f_boxes

    def f_boxes(self, parameters, time_dim=12, use_memmap=False):
        calculation_function = lambda p: self.f_boxes_calculate(p, time_dim=time_dim)
        return self.cache.get_value(parameters, self.f_boxes_cache_filename.format(time_dim), calculation_function, derivative_used=False, save_also_txt=False, use_memmap=use_memmap)



    def df_boxes_calculate(self, parameters, time_dim=12):
        logger.debug('Calculating new model df_boxes with time dim {} for {}.'.format(time_dim, self))
        df_boxes = self.model.df_boxes(parameters, time_dim_desired=time_dim, spinup_options=self.spinup_options, time_step=self.time_step, accuracy_order=self.df_accuracy_order)
        df_boxes = np.asanyarray(df_boxes)
        for i in range(1, df_boxes.ndim-1):
            df_boxes = np.swapaxes(df_boxes, i, i+1)
        return df_boxes

    def df_boxes(self, parameters, time_dim=12, use_memmap=False, as_shared_array=False):
        calculation_function = lambda p: self.df_boxes_calculate(p, time_dim=time_dim)
        return self.cache.get_value(parameters, self.df_boxes_cache_filename.format(time_dim), calculation_function, derivative_used=True, save_also_txt=False, use_memmap=use_memmap, as_shared_array=as_shared_array)



    def F_calculate(self, parameters):
        raise NotImplementedError("Please implement this method")

    def F(self, parameters):
        return self.cache.get_value(parameters, self.F_cache_filename, self.F_calculate, derivative_used=False, save_also_txt=False)

    def DF_calculate(self, parameters):
        raise NotImplementedError("Please implement this method")

    def DF(self, parameters):
        return self.cache.get_value(parameters, self.DF_cache_filename, self.DF_calculate, derivative_used=True, save_also_txt=False)



    ## deviation

    @property
    def deviations_calculate(self):
        raise NotImplementedError("Please implement this method")

    @property
    def deviations(self):
        return self.memory_cache.get_value('deviations', lambda: self.deviations_calculate)

    @property
    def inverse_deviations(self):
        return self.memory_cache.get_value('inverse_deviations', lambda: 1 / self.deviations)

    @property
    def variances(self):
        return self.memory_cache.get_value('variances', lambda: self.deviations**2)

    @property
    def inverse_variances(self):
        return self.memory_cache.get_value('inverse_variances', lambda: 1 / self.variances)

    @property
    def average_variance(self):
        return self.memory_cache.get_value('average_variance', lambda: self.variances.mean())

    @property
    def inverse_average_variance(self):
        return self.memory_cache.get_value('inverse_average_variance', lambda: 1 / self.average_variance)



    ## deviation boxes

    def deviations_boxes(self, time_dim=12, as_shared_array=False):
        return self.memory_cache.get_value('deviations_boxes_{}'.format(time_dim), lambda: measurements.all.pw.values.deviation_TMM(t_dim=time_dim), as_shared_array=as_shared_array)

    def inverse_deviations_boxes(self, time_dim=12, as_shared_array=False):
        return self.memory_cache.get_value('inverse_deviations_boxes_{}'.format(time_dim), lambda: 1 / self.deviations_boxes(time_dim=time_dim), as_shared_array=as_shared_array)


    ## results

    @property
    def results_calculate(self):
        raise NotImplementedError("Please implement this method")

    @property
    def results(self):
        return self.memory_cache.get_value('results', lambda: self.results_calculate)



class WOA(DataBase):

    def __init__(self, spinup_options=DEFAULT_SPINUP_OPTIONS, time_step=1, df_accuracy_order=2, job_setup=None, cache_dirname=None):
        ## super constructor
        from .constants import F_WOA_CACHE_FILENAME, DF_WOA_CACHE_FILENAME
        super().__init__(spinup_options, time_step=time_step, df_accuracy_order=df_accuracy_order, job_setup=job_setup, F_cache_filename=F_WOA_CACHE_FILENAME, DF_cache_filename=DF_WOA_CACHE_FILENAME)

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

    @property
    def deviations_calculate(self):
        return self._get_data_with_annual_averaged(self.mean_deviations_boxes, annual_factor=1/12)


    ## measurements

    @property
    def mask(self):
        return self.memory_cache.get_value('mask', lambda: measurements.all.box.data.nobs() > 0)

    @property
    def results_boxes(self):
        return self.memory_cache.get_value('results_boxes', lambda: measurements.all.box.data.means())

    @property
    def results_calculate(self):
        return self._get_data_with_annual_averaged(self.results_boxes)

    @property
    def m(self):
        return self.memory_cache.get_value('m', lambda: len(self.results))



    ## diff

    def diff_boxes(self, parameters, normalize_with_deviation=False, no_data_value=np.inf):
        results_boxes = self.results_boxes
        results_boxes[np.logical_not(self.mask)] = no_data_value
        diff = results_boxes - self.f_boxes(parameters)
        if normalize_with_deviation:
            diff = diff / self.mean_deviations_boxes
        return diff




class WOD(DataBase):

    def __init__(self, spinup_options=DEFAULT_SPINUP_OPTIONS, time_step=1, df_accuracy_order=2, job_setup=None, cache_dirname=None):
        from .constants import F_WOD_CACHE_FILENAME, DF_WOD_CACHE_FILENAME
        super().__init__(spinup_options, time_step=time_step, df_accuracy_order=df_accuracy_order, job_setup=job_setup, F_cache_filename=F_WOD_CACHE_FILENAME, DF_cache_filename=DF_WOD_CACHE_FILENAME)


    ## model output

    def F_calculate(self, parameters):
        (f_dop, f_po4) = self.model.f_points(parameters, self.points, spinup_options=self.spinup_options, time_step=self.time_step)
        F = np.concatenate([f_dop, f_po4])
        return F


    def DF_calculate(self, parameters):
        (df_dop, df_po4) = self.model.df_points(parameters, self.points, spinup_options=self.spinup_options, time_step=self.time_step, accuracy_order=self.df_accuracy_order)
        DF = np.concatenate([df_dop, df_po4], axis=-1)
        DF = np.swapaxes(DF, 0, 1)
        return DF


    def F_dop(self, parameters):
        F = self.F(parameters)
        return F[:self.m_dop]


    def F_po4(self, parameters):
        F = self.F(parameters)
        return F[self.m_dop:]


    ## deviation

    @property
    def deviations_calculate(self):
        (deviation_dop, deviation_po4) = measurements.all.pw.values.deviation()
        deviations = np.concatenate([deviation_dop, deviation_po4])
        assert len(deviations) == self.m
        return deviations


    ## measurements

    @property
    def points_calculate(self):
        [[points_dop, points_po4], [results_dop, results_po4]] = measurements.all.pw.values.points_and_results()
        return [points_dop, points_po4]

    @property
    def points(self):
        return self.memory_cache.get_value('points', lambda: self.points_calculate)


    @property
    def results_calculate(self):
        [[points_dop, points_po4], [results_dop, results_po4]] = measurements.all.pw.values.points_and_results()
        results = np.concatenate([results_dop, results_po4])
        assert len(results) == self.m
        return results


    @property
    def m_dop(self):
        return self.memory_cache.get_value('m_dop', lambda: len(self.points[0]))

    @property
    def m_po4(self):
        return self.memory_cache.get_value('m_po4', lambda: len(self.points[1]))

    @property
    def m(self):
        return self.memory_cache.get_value('m', lambda: self.m_dop + self.m_po4)


    ## correlation matrix

    def correlation_matrix_calculate(self, min_values, max_year_diff=float('inf')):
        return measurements.all.pw.correlation.CorrelationMatrix(min_values=min_values, max_year_diff=max_year_diff).correlation_matrix_positive_definite

    def correlation_matrix(self, min_values, max_year_diff=float('inf')):
        return self.memory_cache.get_value('correlation_matrix_{:0>2}_{:0>2}'.format(min_values, max_year_diff), lambda: self.correlation_matrix_calculate(min_values, max_year_diff=max_year_diff))


    def correlation_matrix_cholesky_decomposition_calculate(self, min_values, max_year_diff=float('inf')):
        return measurements.all.pw.correlation.CorrelationMatrix(min_values=min_values, max_year_diff=max_year_diff).correlation_matrix_cholesky_decomposition

    def correlation_matrix_cholesky_decomposition(self, min_values, max_year_diff=float('inf')):
        return self.memory_cache.get_value('correlation_matrix_cholesky_decomposition_{:0>2}_{:0>2}'.format(min_values, max_year_diff), lambda: self.correlation_matrix_cholesky_decomposition_calculate(min_values, max_year_diff=max_year_diff))

    # def correlation_matrix_L_calculate(self, min_values, max_year_diff=1):
    #     return measurements.all.pw.correlation.CorrelationMatrix(min_values=min_values, max_year_diff=max_year_diff).correlation_matrix_cholesky_decomposition[0]
    #
    # def correlation_matrix_L(self, min_values, max_year_diff=1):
    #     return self.memory_cache.get_value('correlation_matrix_L_{:0>2}_{:0>2}'.format(min_values, max_year_diff), lambda: self.correlation_matrix_L_calculate(min_values, max_year_diff=max_year_diff))
    #
    #
    # def correlation_matrix_P_calculate(self, min_values, max_year_diff=1):
    #     return measurements.all.pw.correlation.CorrelationMatrix(min_values=min_values, max_year_diff=max_year_diff)..correlation_matrix_cholesky_decomposition[1]
    #
    # def correlation_matrix_P(self, min_values, max_year_diff=1):
    #     return self.memory_cache.get_value('correlation_matrix_P_{:0>2}_{:0>2}'.format(min_values, max_year_diff), lambda: self.correlation_matrix_P_calculate(min_values, max_year_diff=max_year_diff))



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




#     def project_1(self, values):
#         values_squared = []
#
#         for i in range(len(values)):
#             value = values[i]
#             value_matrix = util.math.matrix.convert_to_matrix(value)
#             value_squared = np.array(value_matrix.T * value_matrix)
#             value_squared = util.math.matrix.convert_matrix_to_array(value_squared)
#             values_squared.append(value_squared)
#
#         values_squared = np.array(values_squared)
#
#         return values_squared
#
#
#
#     def project_2(self, values):
#         values_projected = []
#
#         for i in range(len(values)):
#             value = values[i]
#             value_projected = np.sum(value, axis=0)
#
#             values_projected.append(value_projected)
#
#         values_projected = np.array(values_projected)
#
#         return values_projected

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

#     def project(self, values):
#         values_squared = []
#         values_projected = []
#
#         for i in range(len(values)):
#             value = values[i]
#             value_matrix = util.math.matrix.convert_to_matrix(value)
#             value_squared = np.array(value_matrix.T * value_matrix)
#             value_squared = util.math.matrix.convert_matrix_to_array(value_squared)
#             value_projected = np.sum(value, axis=0)
#
#             values_squared.append(value_squared)
#             values_projected.append(value_projected)
#
#         values_squared = np.array(values_squared)
#         values_projected = np.array(values_projected)
#
#         return (values_squared, values_projected)



#     def product_inverse_covariance_matrix_both_sides(self, factor, correlation_parameters):
#         factor = self.inverse_deviations[:, np.newaxis] * factor
#         product = self.product_inverse_correlation_matrix_both_sides(factor, correlation_parameters)
#
#         return product

#     def product_inverse_correlation_matrix_both_sides(self, factor, correlation_parameters):
#         n = self.m_dop
#         factor_projected = self.project([factor[:n], factor[n:]])
#         product = self.projected_product_inverse_correlation_matrix_both_sides(factor_projected, correlation_parameters)
#
#         return product


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
#         data_map = np.concatenate([data_dop_map[np.newaxis], data_po4_map[np.newaxis]], axis=0)
        data_map = [data_dop_map, data_po4_map]

        return data_map


class OLD_WOD(WOD):
    def __init__(self, spinup_options=DEFAULT_SPINUP_OPTIONS, time_step=1, df_accuracy_order=2, job_setup=None, cache_dirname=None):
        from .constants import F_WOD_CACHE_FILENAME, DF_WOD_CACHE_FILENAME
        F_WOD_CACHE_FILENAME = 'old_' + F_WOD_CACHE_FILENAME
        DF_WOD_CACHE_FILENAME = 'old_' + DF_WOD_CACHE_FILENAME
        DataBase.__init__(self, spinup_options, time_step=time_step, df_accuracy_order=df_accuracy_order, job_setup=job_setup, F_cache_filename=F_WOD_CACHE_FILENAME, DF_cache_filename=DF_WOD_CACHE_FILENAME)


    @property
    def deviations_calculate(self):
        import measurements.dop.pw.deviation
        dop_deviation = measurements.dop.pw.deviation.for_points()
        from measurements.po4.wod.constants import ANALYSIS_DIR
        po4_deviation = np.load(ANALYSIS_DIR+'/old/deviation/measurement_deviations_interpolation_52_(0.1,2,0.2).npy')
        return np.concatenate([dop_deviation, po4_deviation])

    @property
    def points_calculate(self):
        import measurements.dop.pw.data
        (dop_points, dop_values) = measurements.dop.pw.data.points_and_values()
        from measurements.po4.wod.constants import ANALYSIS_DIR
        po4_points = np.load(ANALYSIS_DIR+'/old/measurement_points.npy')
        return [dop_points, po4_points]

    @property
    def results_calculate(self):
        import measurements.dop.pw.data
        (dop_points, dop_results) = measurements.dop.pw.data.points_and_values()
        from measurements.po4.wod.constants import ANALYSIS_DIR
        po4_results = np.load(ANALYSIS_DIR+'/old/measurement_results.npy')
        return np.concatenate([dop_results, po4_results])




def init_data_base(data_kind, spinup_options=DEFAULT_SPINUP_OPTIONS, time_step=1, df_accuracy_order=2, job_setup=None):
    if data_kind.upper() == 'WOA':
        data_base_class = WOA
    elif data_kind.upper() == 'WOD':
        data_base_class = WOD
    elif data_kind.upper() == 'OLD_WOD':
        data_base_class = OLD_WOD
    else:
        raise ValueError('Data_kind {} unknown. Must be "WOA" or "WOD".'.format(data_kind))

    data_base = data_base_class(spinup_options, time_step, df_accuracy_order=df_accuracy_order, job_setup=job_setup)

    return data_base




# class Family:
# 
#     def __init__(self, main_member_class, member_classes, data_kind, spinup_options, time_step=1, df_accuracy_order=2, job_setup=None):
# 
#         logger.debug('Initiating cost function family for data kind {} with main member {} and members {}.'.format(data_kind, main_member_class.__name__, list(map(lambda x: x.__name__, member_classes))))
# 
#         if main_member_class not in member_classes:
#             raise ValueError('The main member class has to be in {}, but its {}.'.format(member_classes__name__, main_member_class))
# 
#         main_member = main_member_class(data_kind, spinup_options, time_step=time_step, df_accuracy_order=df_accuracy_order, job_setup=job_setup)
#         self.main_member = main_member
# 
#         family = []
#         for member_class in member_classes:
#             if member_class is not main_member_class:
#                 member = member_class(data_kind, spinup_options, time_step=time_step, df_accuracy_order=df_accuracy_order, job_setup=job_setup)
#                 member.data_base = main_member.data_base
#                 family.append(member)
# 
#         self.family = family
# 
# 
#     def get_function_value(self, function):
#         assert callable(function)
# 
#         value = function(self.main_member)
#         for member in self.family:
#             function(member)
# 
#         return value



class Family():
    
    member_classes = {}
    
    # member_classes = {'WOA': [(OLS, [{}]), (WLS, [{}]), (LWLS, [{}])], 'WOD': [(OLS, [{}]), (WLS, [{}]), (LWLS, [{}]), (GLS, [{'correlation_min_values': a, 'correlation_max_year_diff': float('inf')} for correlation_min_values in (30, 35, 40)])]}

    def __init__(self, **cf_kargs):
        ## chose member classes
        data_kind = cf_kargs['data_kind'].upper()
        try:
            member_classes_list = self.member_classes[data_kind]
        except KeyError:
            raise ValueError('Data_kind {} unknown. Must be in {}.'.format(data_kind, list(self.member_classes.keys())))

        ## init members
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

        value = function(self.main_member)
        for member in self.family:
            function(member)

        return value