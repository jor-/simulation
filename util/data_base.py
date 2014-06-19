import logging
import numpy as np

import measurements.all.woa.data
import measurements.all.pw.data
import ndop.util.value_cache
from ndop.model.eval import Model

import util.math.matrix



class Data_Base:
    
    def __init__(self, years, tolerance=0, combination='or', time_step=1, df_accuracy_order=2, job_setup=None, F_cache_filename=None, DF_cache_filename=None):
        from .constants import CACHE_DIRNAME, F_ALL_CACHE_FILENAME, DF_ALL_CACHE_FILENAME
        
        logging.debug('Initiating {} with {} years, {} tolerance, combination "{}", time step {}, df_accuracy_order {}, job_setup {}, F_cache_filename {} and DF_cache_filename {}.'.format(self.__class__.__name__, years, tolerance, combination, time_step, df_accuracy_order, job_setup, F_cache_filename, DF_cache_filename))
        
        self.years = years
        self.tolerance = tolerance
        self.combination = combination
        self.time_step = time_step
        self.df_accuracy_order = df_accuracy_order
        
        self.cache = ndop.util.value_cache.Cache(years, tolerance, combination, time_step, df_accuracy_order=df_accuracy_order, cache_dirname=CACHE_DIRNAME)
        self.f_all_cache_filename = F_ALL_CACHE_FILENAME
        self.df_all_cache_filename = DF_ALL_CACHE_FILENAME
        self.F_cache_filename = F_cache_filename
        self.DF_cache_filename = DF_cache_filename
        
        if job_setup is None:
            job_setup = {}
        try:
            job_setup['name']
        except KeyError:
            job_setup['name'] = self.__class__.__name__
        self.model = Model(job_setup)
        
        self.last_parameters_f = None
        self.last_parameters_df = None
        self.last_parameters_f_all = None
        self.last_parameters_df_all = None
        self.last_F = None
        self.last_DF = None
        self.last_f_all = None
        self.last_df_all = None
    
    
    ## access to model
    @property
    def inverse_variances_all(self):
        try:
            return self._inverse_variances_all
        except AttributeError:
            self._inverse_variances_all = measurements.all.woa.data.nobs() / measurements.all.woa.data.varis()
            return self._inverse_variances_all
    
    
    
    def f_all_calculate(self, parameters):
        logging.debug('Calculating new model f_all for {}.'.format(self.__class__.__name__))
        f_all = self.model.f_all(parameters, time_dim_desired=12, years=self.years, tolerance=self.tolerance, combination=self.combination, time_step=self.time_step)
        f_all = np.asanyarray(f_all)
        return f_all
    
    def f_all(self, parameters):
        if self.last_f_all is not None and all(parameters == self.last_parameters_f_all):
            logging.debug('Returning cached f_all for {}.'.format(self.__class__.__name__))
            f_all = self.last_f_all
        else:
            if self.f_all_cache_filename is not None:
                f_all = self.cache.get_value(parameters, self.f_all_cache_filename, self.f_all_calculate, derivative_used=False, save_also_txt=False)
            else:
                f_all = self.f_all_calculate(parameters)
            
            self.last_parameters_f_all = parameters
            self.last_f_all = f_all
        
        logging.debug('Returning f_all with shape {} for {}.'.format(f_all.shape, self.__class__.__name__))
        return f_all
    
    
    
    def df_all_calculate(self, parameters):
        logging.debug('Calculating new df_all for {}.'.format(self.__class__.__name__))
        df_all = self.model.df_all(parameters, time_dim_desired=12, years=self.years, tolerance=self.tolerance, combination=self.combination, time_step=self.time_step, accuracy_order=self.df_accuracy_order)
        df_all = np.asanyarray(df_all)
        for i in range(1, df_all.ndim-1):
            df_all = np.swapaxes(df_all, i, i+1)
        return df_all
    
    def df_all(self, parameters):
        if self.last_df_all is not None and all(parameters == self.last_parameters_df_all):
            logging.debug('Returning cached df_all for {}.'.format(self.__class__.__name__))
            df_all = self.last_df_all
        else:
            if self.df_all_cache_filename is not None:
                df_all = self.cache.get_value(parameters, self.df_all_cache_filename, self.df_all_calculate, derivative_used=True, save_also_txt=False)
            else:
                df_all = self.df_all_calculate(parameters)
            
            self.last_parameters_df_all = parameters
            self.last_df_all = df_all
        
        self.DF_used = True
        
        logging.debug('Returning df_all with shape {} for {}.'.format(df_all.shape, self.__class__.__name__))
        return df_all
    
    
    
    def F_calculate(self, parameters):
        raise NotImplementedError("Please implement this method")
    
    def F(self, parameters):
        if self.last_F is not None and all(parameters == self.last_parameters_F):
            logging.debug('Returning cached model f for {}.'.format(self.__class__.__name__))
            F = self.last_F
        else:
            logging.debug('Calculating new model f for {}.'.format(self.__class__.__name__))
            
            
            if self.F_cache_filename is not None:
                F = self.cache.get_value(parameters, self.F_cache_filename, self.F_calculate, derivative_used=False, save_also_txt=False)
            else:
                F = self.F_calculate(parameters)
                
            self.last_parameters_F = parameters
            self.last_F = F
        
        logging.debug('Returning F with shape {} for {}.'.format(F.shape, self.__class__.__name__))
        return F
    
    def DF_calculate(self, parameters):
        raise NotImplementedError("Please implement this method")
    
    def DF(self, parameters):
        if self.last_DF is not None and all(parameters == self.last_parameters_DF):
            logging.debug('Returning cached model df for {}.'.format(self.__class__.__name__))
            DF = self.last_DF
        else:
            logging.debug('Calculating new model df for {}.'.format(self.__class__.__name__))
            
            if self.DF_cache_filename is not None:
                DF = self.cache.get_value(parameters, self.DF_cache_filename, self.DF_calculate, derivative_used=True, save_also_txt=False)
            else:
                DF = self.DF_calculate(parameters)
            
            self.last_parameters_DF = parameters
            self.last_DF = DF
        
        self.DF_used = True
        
        logging.debug('Returning DF with shape {} for {}.'.format(DF.shape, self.__class__.__name__))
        return DF
    
    


class WOA(Data_Base):
    
    def __init__(self, years, tolerance=0, combination='or', time_step=1, df_accuracy_order=2, job_setup=None, cache_dirname=None):
        from .constants import F_WOA_CACHE_FILENAME, DF_WOA_CACHE_FILENAME
        super().__init__(years, tolerance=tolerance, combination=combination, time_step=time_step, df_accuracy_order=df_accuracy_order, job_setup=job_setup, F_cache_filename=F_WOA_CACHE_FILENAME, DF_cache_filename=DF_WOA_CACHE_FILENAME)
    
    
    ## measurements
    
    @property
    def mask(self):
        try:
            return self._mask
        except AttributeError:
            self._mask = measurements.all.woa.data.nobs() > 0
            self._m_dop = (self._mask[0]).sum()
            self._m_po4 = (self._mask[1]).sum()
            return self._mask 
    
    @property
    def results_all(self):
        try:
            return self._results_all
        except AttributeError:
            self._results_all = measurements.all.woa.data.means()
            return self._results_all
    
    @property
    def results(self):
        try:
            return self._results
        except AttributeError:
            self._results = self.results_all[self.mask]
            return self._results
    
    @property
    def m_dop(self):
        try:
            return self._m_dop
        except AttributeError:
            self.mask
            return self._m_dop
    
    @property
    def m_po4(self):
        try:
            return self._m_po4
        except AttributeError:
            self.mask
            return self._m_po4
    
    @property
    def m(self):
        try:
            return self._m
        except AttributeError:
            self._m = len(self.results)
            return self._m
    
    
    ## devitation
    
    @property
    def inverse_variances(self):
        try:
            return self._inverse_variances
        except AttributeError:
            self._inverse_variances = self.inverse_variances_all[self.mask]
            return self._inverse_variances
    
    @property
    def average_inverse_variance(self):
        try:
            return self._average_inverse_variance
        except AttributeError:
            self._average_inverse_variance = self.inverse_variances.mean()
            return self._average_inverse_variance
    
    
    ## model output
    
    def F_calculate(self, parameters):
        f_all = self.f_all(parameters)
        F = f_all[self.mask]
        return F
    
    
    def DF_calculate(self, parameters):
        df_all = self.df_all(parameters)
        DF = df_all[self.mask]
        return DF
    
    
    ## diff
    
    def diff_all(self, parameters, no_data_value=np.inf):
        results_all = self.results_all
        results_all[np.logical_not(self.mask)] = no_data_value
        diff = results_all - self.f_all(parameters)
        return diff
    
    def diff_all_dop(self, parameters):
        diff = self.diff_all(parameters)
        return diff[0]
    
    def diff_all_po4(self, parameters):
        diff = self.diff_all(parameters)
        return diff[1]

   


class WOD(Data_Base):
    
    def __init__(self, years, tolerance=0, combination='or', time_step=1, df_accuracy_order=2, job_setup=None, cache_dirname=None):
        from .constants import F_WOD_CACHE_FILENAME, DF_WOD_CACHE_FILENAME
        super().__init__(years, tolerance=tolerance, combination=combination, time_step=time_step, df_accuracy_order=df_accuracy_order, job_setup=job_setup, F_cache_filename=F_WOD_CACHE_FILENAME, DF_cache_filename=DF_WOD_CACHE_FILENAME)
    
    
    ## measurements
    
    @property
    def points(self):
        try:
            return self._points
        except AttributeError:
            points, results = measurements.all.pw.data.get_points_and_values()
            [[points_dop, points_po4], [results_dop, results_po4]] = measurements.all.pw.data.get_points_and_values()
#             self._points = np.concatenate([points_dop, points_po4])
            self._points = [points_dop, points_po4]
            self._results = np.concatenate([results_dop, results_po4])
            self._m_dop = len(results_dop)
            self._m_po4 = len(results_po4)
            return self._points
    
    @property
    def results(self):
        try:
            return self._results
        except AttributeError:
            self.points
            return self._results
    
    @property
    def m_dop(self):
        try:
            return self._m_dop
        except AttributeError:
            self.points
            return self._m_dop
    
    @property
    def m_po4(self):
        try:
            return self._m_po4
        except AttributeError:
            self.points
            return self._m_po4
    
    @property
    def m(self):
        try:
            return self._m
        except AttributeError:
            self._m = self.m_dop + self.m_po4
            return self._m
    
    
    ## devitation
    
    @property
    def inverse_deviations(self):
        try:
            return self._inverse_deviations
        except AttributeError:
            (deviation_dop, deviation_po4) = measurements.all.pw.data.get_deviation()
            self._inverse_deviations = 1 / np.concatenate([deviation_dop, deviation_po4])
            return self._inverse_deviations
    
    @property
    def inverse_variances(self):
        try:
            return self._inverse_variances
        except AttributeError:
            self._inverse_variances = self.inverse_deviations**(-2)
            return self._inverse_variances
    
    @property
    def average_inverse_variance(self):
        try:
            return self._average_inverse_variance
        except AttributeError:
            self._average_inverse_variance = self.inverse_variances.mean()
            return self._average_inverse_variance
    
    
    ## correlation methods
    
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
#     
#     
#     def product_inverse_covariance_matrix_both_sides(self, factor, correlation_parameters):
#         ## get correlation parameters
#         [a, b, c] = correlation_parameters
#         n = self.m_dop
#         m = self.m_po4
#         
#         ## check regularity
#         self.check_regularity(correlation_parameters)
#         
#         ## calculate first product part
#         factor_deviation_weighted = self.inverse_deviations[:, np.newaxis] * factor
#         factor_deviation_weighted_matrix = np.matrix(factor_deviation_weighted)
#         assert factor_deviation_weighted.shape == factor.shape
#         
#         factor_deviation_and_correlation_weighted = factor_deviation_weighted.copy()
#         factor_deviation_and_correlation_weighted[:n] *= 1 / (1-a)
#         factor_deviation_and_correlation_weighted[n:] *= 1 / (1-b)
#         factor_deviation_and_correlation_weighted_matrix = np.matrix(factor_deviation_and_correlation_weighted)
#         
#         product = factor_deviation_weighted_matrix.T * factor_deviation_and_correlation_weighted_matrix
#         
#         
#         ## calculate core matrix (2x2)
#         A = np.matrix([[a, c], [c, b]])
#         W = np.matrix([[1-a, 0], [0, 1-b]])
#         D = np.matrix([[n, 0], [0, m]])
#         H = W + D * A
#         core = W.I * A * H.I
#         
#         
#         ## create projection matrix (2xn+m)
#         projection_matrix  = np.zeros([2, n+m])
#         projection_matrix[0, :n] = 1
#         projection_matrix[1, n:] = 1
#         projection_matrix = np.matrix(projection_matrix)
#         
#         
#         ## calculate second product part
#         factor_deviation_weighted_projected_matrix = projection_matrix * factor_deviation_weighted_matrix
#         product += factor_deviation_weighted_projected_matrix.T * core * factor_deviation_weighted_projected_matrix
#         assert product.ndim == 2
#         assert product.shape == (factor.shape[1], factor.shape[1])
#         
#         product = np.array(product)
#         if product.size == 1:
#             product = product[0, 0]
#         
#         return product
    
    
    def project(self, values):
        values_squared = []
        values_projected = []
        
        for i in range(len(values)):
            value = values[i]
            value_matrix = util.math.matrix.convert_to_matrix(value)
            value_squared = np.array(value_matrix.T * value_matrix)
            value_squared = util.math.matrix.convert_matrix_to_array(value_squared)
            value_projected = np.sum(value, axis=0)
            
            values_squared.append(value_squared)
            values_projected.append(values_projected)
        
#         if len(values_squared) == 1:
#             values_squared = values_squared[0]
#         if len(values_projected) == 1:
#             values_squared = values_projected[0]
        
        return (values_squared, values_projected)
    
    
#     def product_inverse_covariance_matrix_both_sides(self, factor, correlation_parameters):
#         ## get correlation parameters
#         n = self.m_dop
#         
#         ## calculate factor.T * factor
#         factor = self.inverse_deviations[:, np.newaxis] * factor
#         factor_matrix = np.matrix(factor)
#         factor_squared = [factor_matrix[:n].T * factor_matrix[:n], factor_matrix[n:].T * factor_matrix[n:]]
#         
#         ## create projection matrix (2xn+m)
#         projection_matrix  = np.zeros([2, n+m])
#         projection_matrix[0, :n] = 1
#         projection_matrix[1, n:] = 1
#         projection_matrix = np.matrix(projection_matrix)
#         
#         ## project factor
#         factor_projected = projection_matrix * factor_matrix
#         
#         ## calculate product
#         product = self.projected_product_inverse_correlation_matrix_both_sides(factor_squared, factor_projected, correlation_parameters)
#         
#         return product
    
    
    def product_inverse_covariance_matrix_both_sides(self, factor, correlation_parameters):
        factor = self.inverse_deviations[:, np.newaxis] * factor
        product = self.product_inverse_correlation_matrix_both_sides(factor, correlation_parameters)
        
        return product
    
    def product_inverse_correlation_matrix_both_sides(self, factor, correlation_parameters):
        n = self.m_dop
        
        (factor_squared, factor_projected) = self.project([factor[:n], factor[n:]])
        product = self.projected_product_inverse_correlation_matrix_both_sides(factor_squared, factor_projected, correlation_parameters)
        
        return product
    
    
    def projected_product_inverse_correlation_matrix_both_sides(self, factor_squared, factor_projected, correlation_parameters):
        assert len(factor_squared) == 2
        assert len(factor_projected) == 2
        
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
        core = W.I * A * H.I
        
        ## calculate second product part
        factor_projected_matrix = util.math.matrix.convert_to_matrix(factor_projected)
        product += factor_projected_matrix.T * core * factor_projected_matrix
        
        ## return product
        product = np.array(product)
        if product.size == 1:
            product = product[0, 0]
        
        return product
    
    
    ## model output
    
    def F_calculate(self, parameters):
        (f_dop, f_po4) = self.model.f_points(parameters, self.points, years=self.years, tolerance=self.tolerance, combination=self.combination, time_step=self.time_step)
        F = np.concatenate([f_dop, f_po4])
        return F
    
    
    def DF_calculate(self, parameters):
        (df_dop, df_po4) = self.model.df_points(parameters, self.points, years=self.years, tolerance=self.tolerance, combination=self.combination, time_step=self.time_step, accuracy_order=self.df_accuracy_order)
        DF = np.concatenate([df_dop, df_po4], axis=-1)
        DF = np.swapaxes(DF, 0, 1)
        return DF
    
    
    def F_dop(self, parameters):
        F = self.F(parameters)
        return F[:self.m_dop]
    
    
    def F_po4(self, parameters):
        F = self.F(parameters)
        return F[self.m_dop:]
    
    
    ## diff
    
    def diff(self, parameters):
        diff = self.results - self.F(parameters)
        return diff
    
    def diff_dop(self, parameters):
        diff = self.diff(parameters)
        return diff[:self.m_dop]
    
    def diff_po4(self, parameters):
        diff = self.diff(parameters)
        return diff[self.m_dop:]


# class All():
#     
#     def __init__(self, years, tolerance=0, combination='or', time_step=1, df_accuracy_order=2, job_setup=None, cache_dirname=None):
# #         self.configs = dict(years=years, tolerance=tolerance, combination=combination, time_step=time_step, df_accuracy_order=df_accuracy_order, job_setup=job_setup)
#         self.configs = dict(((k, eval(k)) for k in ('years', 'tolerance', 'combination', 'time_step', 'df_accuracy_order', 'job_setup', 'cache_dirname')))
#         self.data_bases = {}
#     
#     
#     def __getitem__(self, data_kind):
#         data_kind = data_kind.upper()
#         try:
#             self.data_bases[data_kind]
#         except KeyError as e:
#             if data_kind == 'WOA':
#                 data_base_class = WOA
#             elif data_kind == 'WOD':
#                 data_base_class = WOD
#             else:
#                 raise ValueError('Data_kind {} unknown. Must be "WOA" or "WOD".'.format(data_kind))
#             self.data_bases[data_kind] = data_base_class(**self.configs)
#         return self.data_bases[data_kind]


def init_data_base(data_kind, years=1, tolerance=0, combination='or', time_step=1, df_accuracy_order=2, job_setup=None):
    if data_kind.upper() == 'WOA':
        data_base_class = WOA
    elif data_kind.upper() == 'WOD':
        data_base_class = WOD
    else:
        raise ValueError('Data_kind {} unknown. Must be "WOA" or "WOD".'.format(data_kind))
    
    data_base = data_base_class(years, tolerance, combination, time_step, df_accuracy_order=df_accuracy_order, job_setup=job_setup)
    
    return data_base




class Family:
    
    def __init__(self, main_member_class, member_classes, data_kind, years, tolerance=0, combination='or', time_step=1, df_accuracy_order=2, job_setup=None):
        
        logging.debug('Initiating cost function family for data kind {} with main member {} and members {}.'.format(data_kind, main_member_class.__name__, list(map(lambda x: x.__name__, member_classes))))
        
        if main_member_class not in member_classes:
            raise ValueError('The main member class has to be in {}, but its {}.'.format(member_classes__name__, main_member_class))
        
        main_member = main_member_class(data_kind, years, tolerance=tolerance, combination=combination, time_step=time_step, df_accuracy_order=df_accuracy_order, job_setup=job_setup)
        self.main_member = main_member
        
        family = []
        for member_class in member_classes:
            if member_class is not main_member_class:
                member = member_class(data_kind, years, tolerance=tolerance, combination=combination, time_step=time_step, df_accuracy_order=df_accuracy_order, job_setup=job_setup)
                member.data_base = main_member.data_base
                family.append(member)
        
        self.family = family
    
    
    def get_function_value(self, function):
        assert callable(function)
        
        value = function(self.main_member)
        for member in self.family:
            function(member)
        
        return value