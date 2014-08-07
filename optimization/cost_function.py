import os.path
import numpy as np
import logging

import ndop.util.value_cache
import ndop.util.data_base

import util.math.optimize
from util.math.matrix import SingularMatrixError

from .constants import COST_FUNCTIONS_DIRNAME, COST_FUNCTION_F_FILENAME, COST_FUNCTION_DF_FILENAME, COST_FUNCTION_F_NORMALIZED_FILENAME, COST_FUNCTION_CORRELATION_PARAMETER_FILENAME, COST_FUNCTION_NODES_SETUP_SPINUP, COST_FUNCTION_NODES_SETUP_DERIVATIVE



class Base():
    
    def __init__(self, data_kind, spinup_options, time_step=1, df_accuracy_order=2, job_setup=None):
        cf_kind = self.__class__.__name__
        kind = data_kind + '_' + cf_kind
        
        ## prepare job setup
        if job_setup is None:
            job_setup = {}
        try:
            job_setup['name']
        except KeyError:
            job_setup['name'] = 'O_' + kind
        
        try:
            job_setup['nodes_setup']
        except KeyError:
            try:
                job_setup['spinup']
            except KeyError:
                job_setup['spinup'] = {}
            try:
                job_setup['spinup']['nodes_setup']
            except KeyError:
                job_setup['spinup']['nodes_setup'] = COST_FUNCTION_NODES_SETUP_SPINUP
            try:
                job_setup['derivative']
            except KeyError:
                job_setup['derivative'] = {}
            try:
                job_setup['derivative']['nodes_setup']
            except KeyError:
                job_setup['derivative']['nodes_setup'] = COST_FUNCTION_NODES_SETUP_DERIVATIVE
        
        ## prepare cache and data base
        cache_dirname = os.path.join(COST_FUNCTIONS_DIRNAME, kind)
        self.cache = ndop.util.value_cache.Cache(spinup_options, time_step, df_accuracy_order=df_accuracy_order, cache_dirname=cache_dirname)
        self.data_base = ndop.util.data_base.init_data_base(data_kind, spinup_options, time_step, df_accuracy_order=df_accuracy_order, job_setup=job_setup)
    
    
    def f_calculate(self, parameters):
        raise NotImplementedError("Please implement this method")
    
    def f(self, parameters):
        return self.cache.get_value(parameters, COST_FUNCTION_F_FILENAME, self.f_calculate, derivative_used=False)
    
    def f_normalized_calculate(self, parameters):
        raise NotImplementedError("Please implement this method")
    
    def f_normalized(self, parameters):
        return self.cache.get_value(parameters, COST_FUNCTION_F_NORMALIZED_FILENAME, self.f_normalized_calculate, derivative_used=False)
    
    def df_calculate(self, parameters):
        raise NotImplementedError("Please implement this method")
    
    def df(self, parameters):
        return self.cache.get_value(parameters, COST_FUNCTION_DF_FILENAME, self.df_calculate, derivative_used=True)



class OLS(Base):
    
    def f_calculate(self, parameters):
        F = self.data_base.F(parameters)
        results = self.data_base.results
        
        f = np.sum((results - F)**2)
        
        return f
    
    def f_normalized_calculate(self, parameters):
        f = self.f(parameters)
        m = self.data_base.m
        average_inverse_variance = self.data_base.average_inverse_variance
        
        f_normalized = f * average_inverse_variance / m
        
        return f_normalized
    
    def df_calculate(self, parameters):
        F = self.data_base.F(parameters)
        DF = self.data_base.DF(parameters)
        results = self.data_base.results
        
#         df_factors = results - F
#         
#         p_dim = len(parameters)
#         df = np.empty(p_dim)
#         
#         for i in range(p_dim):
#             df[i] = np.sum(df_factors * DF[i])
#         
#         df *= - 2
        
        df_factors = results - F
        df = - 2 * np.sum(df_factors[:, np.newaxis] * DF, axis=0)
        
        return df



class WLS(Base):
    
    def f_calculate(self, parameters):
        F = self.data_base.F(parameters)
        results = self.data_base.results
        inverse_variances = self.data_base.inverse_variances
        
        f = np.sum((results - F)**2 * inverse_variances)
        
        return f
    
    def f_normalized_calculate(self, parameters):
        f = self.f(parameters)
        m = self.data_base.m
        
        f_normalized = f / m
        
        return f_normalized
    
    def df_calculate(self, parameters):
        F = self.data_base.F(parameters)
        DF = self.data_base.DF(parameters)
        results = self.data_base.results
        inverse_variances = self.data_base.inverse_variances
        
#         df_factors = (results - F) * inverse_variances
#         p_dim = len(parameters)
#         df = np.empty(p_dim)
#         
#         for i in range(p_dim):
#             df[i] = np.sum(df_factors * DF[i])
#         
#         df *= - 2
        
        df_factors = (results - F) * inverse_variances
        df = - 2 * np.sum(df_factors[:, np.newaxis] * DF, axis=0)
        
        return df



class GLS(Base):
    
    def __init__(self, data_kind, spinup_options, time_step=1, df_accuracy_order=2, job_setup=None):
        ## super init
        if data_kind.upper() != 'WOD':
            raise ValueError('Data_kind {} unknown. Must be "WOD".'.format(data_kind))
        super().__init__(data_kind, spinup_options, time_step=1, df_accuracy_order=2, job_setup=job_setup)
        
        ## setup correlation bounds and last correlations
        dim = 3
#         dim = 2
        upper_bound = 1 - 10**(-4)
        lower_bound = 0
        bounds = ((lower_bound, upper_bound),) * dim
        self.correlation_parameters_bounds = bounds
        
        ineq_factor = 4/5
        self.correlation_parameters_ineq_constraint = lambda x: ineq_factor * x[0] * x[1] - x[2]**2 
        self.correlation_parameters_ineq_constraints_jac = lambda x: (ineq_factor * x[1], ineq_factor * x[0], -2 * x[2])
        
        last = np.array([0.0]*dim)
        self.last_correlation_parameters = last
    
    
    def f_calculate_with_diff_and_cp(self, diff_squared, diff_projected, correlation_parameters):
        ## check input
        if not np.all(np.logical_and(correlation_parameters > -1, correlation_parameters < 1)):
            raise ValueError('Each correlation parameter have to be in (-1, 1), but they are {}.'.format(correlation_parameters))
        
        if len(correlation_parameters) == 2:
            correlation_parameters = list(correlation_parameters) + [0]
        
        ## ln det
        try:
            ln_det = self.data_base.ln_det_correlation_matrix(correlation_parameters)
        except SingularMatrixError:
            warnings.warn('Correlation matrix is singular for m={}, n={}, a={}, b={} and c={}.'.format(m, n, a, b, c))
            return np.inf
        
        ## product value
        product_value = self.data_base.projected_product_inverse_correlation_matrix_both_sides(diff_squared, diff_projected, correlation_parameters)
        
        ## calculate function value
        f = ln_det + product_value
        
#         logging.debug('Returning function value for correlation parameters {} with ln_det={} and product_value={}.'.format(correlation_parameters, ln_det, product_value))
        
        return f
    
    
    def f_calculate_with_diff(self, diff_squared, diff_projected):
        ## optimize correlation parameters
        f = lambda correlation_parameters: self.f_calculate_with_diff_and_cp(diff_squared, diff_projected, correlation_parameters)
        last_correlation_parameters = self.last_correlation_parameters
        
        (opt_correlation_parameters, opt_f) = util.math.optimize.minimize(f, last_correlation_parameters, bounds=self.correlation_parameters_bounds, ineq_constraints=self.correlation_parameters_ineq_constraint, ineq_constraints_jac=self.correlation_parameters_ineq_constraints_jac, global_method='basin_hopping', global_iterations=200, global_stepsize=0.05, global_stepsize_update_interval=20)
#         (correlation_parameters_opt, f_opt) = util.math.optimize.minimize(f, last_correlation_parameters, bounds=self.correlation_parameters_bounds, global_method='basin_hopping', global_iterations=200, global_stepsize=0.05, global_stepsize_update_interval=20)
        
        ## save correlation parameters
        self.last_correlation_parameters = correlation_parameters_opt
        
        logging.debug('Returning optimal correlation parameters {} with value {}.'.format(correlation_parameters_opt, f_opt))
        
        return f_opt
    
    
    def f_calculate(self, parameters):
        ## calculate diff_projected and diff_squared
        F = self.data_base.F(parameters)
        results = self.data_base.results
        inverse_deviations = self.data_base.inverse_deviations
        n = self.data_base.m_dop
        
        diff = (results - F) * inverse_deviations
        diff_projected = [np.sum(diff[:n]), np.sum(diff[n:])]
        diff_squared = [np.sum(diff[:n]**2), np.sum(diff[n:]**2)]
        
        ## calculate f
        f = self.f_calculate_with_diff(diff_squared, diff_projected)
        self.cache.save_file(parameters, COST_FUNCTION_CORRELATION_PARAMETER_FILENAME, self.last_correlation_parameters)
        
        return f
    
    
    
    
#     def f_calculate_with_cp(self, parameters, correlation_parameters):
#         ## check input
#         if not np.all(np.logical_and(correlation_parameters > -1, correlation_parameters < 1)):
#             raise ValueError('Each correlation parameter have to be in (-1, 1), but they are {}.'.format(correlation_parameters))
#         
#         ## ln det
#         try:
#             ln_det = self.data_base.ln_det_correlation_matrix(correlation_parameters)
#         except SingularMatrixError:
#             warnings.warn('Correlation matrix is singular for m={}, n={}, a={}, b={} and c={}.'.format(m, n, a, b, c))
#             return np.inf
#         
#         ## product value
#         F = self.data_base.F(parameters)
#         results = self.data_base.results
#         product_value = self.data_base.product_inverse_covariance_matrix_both_sides(results - F)
#         
#         ## calculate function value
#         f = ln_det + product_value
#         
# #         logging.debug('Returning function value for correlation parameters {} with ln_det={} and product_value={}.'.format(correlation_parameters, ln_det, product_value))
#         
#         return f
    
    
#     def f_calculate(self, parameters):
#         
#         ## optimize correlation parameters
#         f = lambda correlation_parameters: self.f_calculate_with_cp(parameters, correlation_parameters)
#         last_correlation_parameters = self.last_correlation_parameters
#         
# #         (opt_correlation_parameters, opt_f) = util.math.optimize.minimize(f, last_correlation_parameters, global_method='basin_hopping', global_iterations=100, bounds=self.correlation_parameters_bounds, ineq_constraints=self.correlation_parameters_ineq_constraint, ineq_constraints_jac=self.correlation_parameters_ineq_constraints_jac)
#         (opt_correlation_parameters, opt_f) = util.math.optimize.minimize(f, last_correlation_parameters, bounds=self.correlation_parameters_bounds, global_method='basin_hopping', global_iterations=200, global_stepsize=0.05, global_stepsize_update_interval=20)
#         
#         ## save correlation parameters
#         self.last_correlation_parameters = opt_correlation_parameters
#         self.data_base.save_file(parameters, COST_FUNCTION_CORRELATION_PARAMETER_FILENAME, self.last_correlation_parameters)
#         
#         logging.debug('Returning optimal correlation parameters {} with value {}.'.format(opt_correlation_parameters, opt_f))
#         
#         return opt_f
    
    def f_normalized_calculate(self, parameters):
        f = self.f(parameters)
        m = self.data_base.m
        
        f_normalized = f / m
        
        return f_normalized
    
    def df_calculate(self, parameters):
        ## calculate diff_projected and diff_squared and its derivatives
        F = self.data_base.F(parameters)
        DF = self.data_base.DF(parameters)
        results = self.data_base.results
        inverse_deviations = self.data_base.inverse_deviations
        n = self.data_base.m_dop
        
        diff = (results - F) * inverse_deviations
        diff_projected = [np.sum(diff[:n]), np.sum(diff[n:])]
        diff_squared = [np.sum(diff[:n]**2), np.sum(diff[n:]**2)]
        
        d_diff = - DF * inverse_deviations[:, np.newaxis]
        p_dim = len(parameters)
        tracer_dim = 2
        d_diff_projected = np.empty([tracer_dim, p_dim])
        d_diff_squared = np.empty([tracer_dim, p_dim])
        for j in range(p_dim):
            d_diff_projected[0, j] = np.sum(d_diff[:n, j])
            d_diff_projected[1, j] = np.sum(d_diff[n:, j])
            d_diff_squared[0, j] = 2 * np.sum(diff[:n] * d_diff[:n, j])
            d_diff_squared[1, j] = 2 * np.sum(diff[n:] * d_diff[n:, j])
        
        
        ## calculate function values and its derivatives
        f_p = self.f_calculate_with_diff(diff_squared, diff_projected)
        d1_f_p = util.math.optimize.finite_differences(lambda diff_squared: self.f_calculate_with_diff(diff_squared, diff_projected), diff_squared, f_x=f_p, bounds=((0, np.inf),)*2, accuracy_order=1)
        d2_f_p = util.math.optimize.finite_differences(lambda diff_projected: self.f_calculate_with_diff(diff_squared, diff_projected), diff_projected, f_x=f_p, bounds=None, accuracy_order=1)
        
        ## compose derivative
        d_f_p = np.matrix(d1_f_p) * np.matrix(d_diff_squared) + np.matrix(d2_f_p) * np.matrix(d_diff_projected)
        d_f_p = np.array(d_f_p.flat)
        
        return d_f_p



class Family(ndop.util.data_base.Family):
    def __init__(self, main_member_class, data_kind, spinup_options, time_step=1, df_accuracy_order=2, job_setup=None):
        
        if data_kind.upper() == 'WOA':
            member_classes = (OLS, WLS)
        elif data_kind.upper() == 'WOD':
            member_classes = (OLS, WLS, GLS)
        else:
            raise ValueError('Data_kind {} unknown. Must be "WOA" or "WOD".'.format(data_kind))
        
        super().__init__(main_member_class, member_classes, data_kind, spinup_options, time_step=time_step, df_accuracy_order=df_accuracy_order, job_setup=job_setup)
    
    def f(self, parameters):
        fun = lambda o: o.f(parameters) 
        value = self.get_function_value(fun)
        self.f_normalized(parameters)
        return value
    
    def f_normalized(self, parameters):
        fun = lambda o: o.f_normalized(parameters) 
        value = self.get_function_value(fun)
        return value
    
    def df(self, parameters):
        fun = lambda o: o.df(parameters) 
        value = self.get_function_value(fun)
        return value
