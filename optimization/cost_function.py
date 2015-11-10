import os.path
import warnings

import numpy as np
import scikits.sparse.cholmod

import ndop.util.value_cache
import ndop.util.data_base

import util.math.optimize.with_scipy
import util.math.finite_differences
import util.math.sparse.create
import util.math.sparse.solve
import util.math.sparse.decompose.with_cholmod
from util.math.matrix import SingularMatrixError

import util.logging
logger = util.logging.logger

from ndop.optimization.constants import COST_FUNCTION_DIRNAME, COST_FUNCTION_F_FILENAME, COST_FUNCTION_DF_FILENAME, COST_FUNCTION_F_NORMALIZED_FILENAME, COST_FUNCTION_CORRELATION_PARAMETER_FILENAME, COST_FUNCTION_NODES_SETUP_SPINUP, COST_FUNCTION_NODES_SETUP_DERIVATIVE, COST_FUNCTION_NODES_SETUP_TRAJECTORY


## Base

class Base():

    def __init__(self, data_kind, spinup_options=None, derivative_options=None, time_step=1, job_setup=None):
        ## save kargs
        self.kargs = {'data_kind':data_kind, 'spinup_options':spinup_options, 'time_step':time_step, 'derivative_options':derivative_options, 'job_setup':job_setup}

        ##
        cf_kind = str(self)

        ## prepare job setup
        if job_setup is None:
            job_setup = {}
        try:
            job_setup['name']
        except KeyError:
            job_setup['name'] = data_kind + '_' + cf_kind

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
            try:
                job_setup['trajectory']
            except KeyError:
                job_setup['trajectory'] = {}
            try:
                job_setup['trajectory']['nodes_setup']
            except KeyError:
                job_setup['trajectory']['nodes_setup'] = COST_FUNCTION_NODES_SETUP_TRAJECTORY

        ## prepare cache and data base
        # self.data_base = ndop.util.data_base.init_data_base(data_kind, spinup_options, time_step, df_accuracy_order=df_accuracy_order, job_setup=job_setup)

        #   self.cache = ndop.util.value_cache.Cache(spinup_options, time_step, df_accuracy_order=df_accuracy_order, cache_dirname=self.cache_dirname, use_memory_cache=True)
        self.data_base = ndop.util.data_base.init_data_base(data_kind, spinup_options=spinup_options, derivative_options=derivative_options, time_step=time_step, job_setup=job_setup)
        self.cache = ndop.util.value_cache.Cache(spinup_options=spinup_options, derivative_options=derivative_options, time_step=time_step, cache_dirname=self.cache_dirname, use_memory_cache=True)


    def __str__(self):
        return self.kind

    @property
    def kind(self):
        return self.__class__.__name__

    @property
    def cache_dirname(self):
        return os.path.join(COST_FUNCTION_DIRNAME, str(self.data_base), self.__class__.__name__)


    ## cost function values

    def f_calculate(self, parameters):
        raise NotImplementedError("Please implement this method")

    def f(self, parameters):
        return self.cache.get_value(parameters, COST_FUNCTION_F_FILENAME, self.f_calculate, derivative_used=False)


    def f_normalized_calculate(self, parameters):
        f = self.f(parameters)
        m = self.data_base.m
        f_normalized = f / m
        return f_normalized

    def f_normalized(self, parameters):
        return self.cache.get_value(parameters, COST_FUNCTION_F_NORMALIZED_FILENAME, self.f_normalized_calculate, derivative_used=False)


    def df_calculate(self, parameters):
        raise NotImplementedError("Please implement this method")

    def df(self, parameters):
        return self.cache.get_value(parameters, COST_FUNCTION_DF_FILENAME, self.df_calculate, derivative_used=True)


    def f_available(self, parameters):
        return self.cache.has_value(parameters, COST_FUNCTION_F_FILENAME)

    def df_available(self, parameters):
        return self.cache.has_value(parameters, COST_FUNCTION_DF_FILENAME)


    ## model and data values

    def model_f(self, parameters):
        return self.data_base.F(parameters)

    def model_df(self, parameters):
        return self.data_base.DF(parameters)

    @property
    def results(self):
        return self.data_base.results



class BaseWeighted(Base):

    @property
    def variances(self):
        return self.data_base.variances

    @property
    def inverse_variances(self):
        return self.data_base.inverse_variances

    @property
    def standard_deviations(self):
        return self.data_base.deviations

    @property
    def inverse_standard_deviations(self):
        return self.data_base.inverse_deviations



class BaseGeneralized(BaseWeighted):

    def __init__(self, *args, correlation_min_values=10, correlation_max_year_diff=float('inf'), **kargs):
        from measurements.constants import CORRELATION_MIN_DIAG_VALUE_POSITIVE_DEFINITE_APPROXIMATION
        self.min_diag_value = CORRELATION_MIN_DIAG_VALUE_POSITIVE_DEFINITE_APPROXIMATION
        
        ## save additional kargs
        self.correlation_min_values = correlation_min_values
        if correlation_max_year_diff is None or correlation_max_year_diff < 0:
            correlation_max_year_diff = float('inf')
        self.correlation_max_year_diff = correlation_max_year_diff

        ## super init
        super().__init__(*args, **kargs)

        ## save additional kargs
        self.kargs['correlation_min_values'] = correlation_min_values
        self.kargs['correlation_max_year_diff'] = correlation_max_year_diff


    def __str__(self):
        return '{}.{}.{}'.format(self.__class__.__name__, self.correlation_min_values, self.correlation_max_year_diff)


    @property
    def cache_dirname(self):
        return os.path.join(COST_FUNCTION_DIRNAME, str(self.data_base), self.__class__.__name__, 'min_values_{}'.format(self.correlation_min_values), 'max_year_diff_{}'.format(self.correlation_max_year_diff), 'min_diag_{:.0e}'.format(self.min_diag_value))


    @property
    def correlation_matrix(self):
        return self.data_base.correlation_matrix(self.correlation_min_values, self.correlation_max_year_diff)

    @property
    def correlation_matrix_cholesky_decomposition(self):
        return self.data_base.correlation_matrix_cholesky_decomposition(self.correlation_min_values, self.correlation_max_year_diff)

    # @property
    # def correlation_matrix_L(self):
    #     return self.data_base.correlation_matrix_L(self.correlation_min_values, self.correlation_max_year_diff)
    #
    # @property
    # def correlation_matrix_P(self):
    #     return self.data_base.correlation_matrix_P(self.correlation_min_values, self.correlation_max_year_diff)


    @property
    def covariance_matrix(self):
        C = self.correlation_matrix
        s = self.standard_deviations
        S = util.math.sparse.create.diag(s)
        C = S * C * S
        return C



class BaseLog(Base):

    def __init__(self, *args, **kargs):
        from .constants import CONCENTRATION_MIN_VALUE
        self.min_value = CONCENTRATION_MIN_VALUE

        super().__init__(*args, **kargs)

    def model_f(self, parameters):
        return np.maximum(super().model_f(parameters), self.min_value)

    def model_df(self, parameters):
        min_mask = super().model_f(parameters) < self.min_value
        df = super().model_df(parameters)
        df[min_mask] = 0
        return df

    @property
    def results(self):
        return np.maximum(super().results, self.min_value)




## Normal distribution

class OLS(Base):

    def f_calculate(self, parameters):
        F = self.model_f(parameters)
        results = self.results

        f = np.sum((F - results)**2)

        return f


    def f_normalized_calculate(self, parameters):
        f_normalized = super().f_normalized_calculate(parameters)
        f_normalized = f_normalized * self.data_base.inverse_average_variance
        return f_normalized


    def df_calculate(self, parameters):
        F = self.model_f(parameters)
        DF = self.model_df(parameters)
        results = self.results

        df_factors = F - results
        df = 2 * np.sum(df_factors[:, np.newaxis] * DF, axis=0)

        return df



class WLS(BaseWeighted):

    def f_calculate(self, parameters):
        F = self.model_f(parameters)
        results = self.results
        inverse_variances = self.inverse_variances

        f = np.sum((F - results)**2 * inverse_variances)

        return f


    def df_calculate(self, parameters):
        F = self.model_f(parameters)
        DF = self.model_df(parameters)
        results = self.results
        inverse_variances = self.inverse_variances

        df_factors = (F - results) * inverse_variances
        df = 2 * np.sum(df_factors[:, np.newaxis] * DF, axis=0)

        return df



class GLS(BaseGeneralized):

    def inv_cov_matrix_mult_residuum_calculate(self, parameters):
        F = self.model_f(parameters)
        results = self.results
        inverse_deviations = self.inverse_standard_deviations

        weighted_residual =  (F - results) * inverse_deviations
        P, L = self.correlation_matrix_cholesky_decomposition

        y = weighted_residual
        x = util.math.sparse.solve.cholesky(L, y, P=P)

        return x


    def inv_cov_matrix_mult_residuum(self, parameters):
        from ndop.optimization.constants import COST_FUNCTION_GLS_PROD_FILENAME
        return self.cache.get_value(parameters, COST_FUNCTION_GLS_PROD_FILENAME, self.inv_cov_matrix_mult_residuum_calculate, derivative_used=False, save_also_txt=False)


    def f_calculate(self, parameters):
        F = self.model_f(parameters)
        results = self.results
        inverse_deviations = self.inverse_standard_deviations

        weighted_residual =  (F - results) * inverse_deviations
        inv_cov_matrix_mult_residuum = self.inv_cov_matrix_mult_residuum(parameters)

        f = np.sum(weighted_residual * inv_cov_matrix_mult_residuum)
        return f


    def df_calculate(self, parameters):
        DF = self.model_df(parameters)
        inverse_deviations = self.inverse_standard_deviations

        df_factors = self.inv_cov_matrix_mult_residuum(parameters) * inverse_deviations

        df = 2 * np.sum(df_factors[:,np.newaxis] * DF, axis=0)
        return df






class GLS_P3(Base):

    # def __init__(self, data_kind, spinup_options, time_step=1, df_accuracy_order=2, job_setup=None):
    def __init__(self, *args, **kargs):
        ## super init
        if data_kind.upper() != 'WOD':
            raise ValueError('Data_kind {} not supported. Must be "WOD".'.format(data_kind))
        super().__init__(*args, **kargs)

        ## setup correlation bounds and last correlations
        self.converted_correlation_parameters_bounds = ((0, 0.99), (0, 0.99), (0, 0.75))

        self.last_correlation_parameters = np.array([0.1, 0.1, 0.001])
        self.singular_function_value = np.finfo(np.float64).max


    def f_calculate_with_diff_and_cp(self, diff_projected, correlation_parameters):
        ## check input
        if not np.all(np.logical_and(correlation_parameters > -1, correlation_parameters < 1)):
            raise ValueError('Each correlation parameter have to be in (-1, 1), but they are {}.'.format(correlation_parameters))

        if len(correlation_parameters) == 2:
            correlation_parameters = list(correlation_parameters) + [0]

        ## ln det
        try:
            ln_det = self.data_base.ln_det_correlation_matrix(correlation_parameters)
        except SingularMatrixError:
            warnings.warn('Correlation matrix is singular for correlation parameters {}.'.format(correlation_parameters))
            return self.singular_function_value

        ## product value
        product_value = self.data_base.projected_product_inverse_correlation_matrix_both_sides(diff_projected, correlation_parameters)

        ## calculate function value
        f = ln_det + product_value

        return f


    def f_calculate_with_diff(self, diff_projected):
        def converted_cp_to_cp(converted_correlation_parameters):
            correlation_parameters = np.copy(converted_correlation_parameters)
            correlation_parameters[2] = (correlation_parameters[0] * correlation_parameters[1] * correlation_parameters[2])**(1/2)
            return correlation_parameters

        def cp_to_converted_cp(correlation_parameters):
            converted_correlation_parameters = np.copy(correlation_parameters)
            if 0 in correlation_parameters[:2]:
                converted_correlation_parameters[2] = 0
            else:
                converted_correlation_parameters[2] = correlation_parameters[2]**2 / (correlation_parameters[0] * correlation_parameters[1])
            return converted_correlation_parameters


        ## optimize correlation parameters
        f = lambda converted_correlation_parameters: self.f_calculate_with_diff_and_cp(diff_projected, converted_cp_to_cp(converted_correlation_parameters))
        last_correlation_parameters = self.last_correlation_parameters

        (converted_opt_correlation_parameters, opt_f) = util.math.optimize.with_scipy.minimize(f, cp_to_converted_cp(self.last_correlation_parameters), bounds=self.converted_correlation_parameters_bounds, global_method='basin_hopping', global_iterations=200, global_stepsize=0.05, global_stepsize_update_interval=20)

        ## save correlation parameters
        self.last_correlation_parameters = converted_cp_to_cp(converted_opt_correlation_parameters)

        logger.debug('Returning optimal correlation parameters {} with value {}.'.format(self.last_correlation_parameters, opt_f))

        return opt_f


    def f_calculate(self, parameters):
        ## calculate diff_projected
        F = self.model_f(parameters)
        results = self.results
        inverse_deviations = self.inverse_standard_deviations
        n = self.data_base.m_dop
        diff = (results - F) * inverse_deviations
#         diff_projected = [np.sum(diff[:n]), np.sum(diff[n:])]
#         diff_squared = [np.sum(diff[:n]**2), np.sum(diff[n:]**2)]
        diff_projected = self.data_base.project(diff, n)

        ## calculate f
        f = self.f_calculate_with_diff(diff_projected)
        self.cache.save_file(parameters, COST_FUNCTION_CORRELATION_PARAMETER_FILENAME, self.last_correlation_parameters)

        return f


    def f_normalized_calculate(self, parameters):
        f = self.f(parameters)
        m = self.data_base.m

        f_normalized = f / m

        return f_normalized


    def df_calculate(self, parameters):
        ## calculate diff_projected and diff_squared and its derivatives
        F = self.model_f(parameters)
        DF = self.model_df(parameters)
        results = self.results
        inverse_deviations = self.inverse_standard_deviations
        n = self.data_base.m_dop

        diff = (results - F) * inverse_deviations
        diff_projected = self.data_base.project(diff, n)
        diff_squared, diff_summed = diff_projected

        d_diff = - DF * inverse_deviations[:, np.newaxis]
        p_dim = len(parameters)
        tracer_dim = 2
        d_diff_summed = np.empty([tracer_dim, p_dim])
        d_diff_squared = np.empty([tracer_dim, p_dim])
        for j in range(p_dim):
            d_diff_summed[0, j] = np.sum(d_diff[:n, j])
            d_diff_summed[1, j] = np.sum(d_diff[n:, j])
            d_diff_squared[0, j] = 2 * np.sum(diff[:n] * d_diff[:n, j])
            d_diff_squared[1, j] = 2 * np.sum(diff[n:] * d_diff[n:, j])


        ## calculate function values and its derivatives
        f_p = self.f_calculate_with_diff(diff_projected)
        d1_f_p = util.math.finite_differences.calculate(lambda diff_squared: self.f_calculate_with_diff([diff_squared, diff_summed]), diff_squared, f_x=f_p, bounds=((0, np.inf),)*2, accuracy_order=1)
        d2_f_p = util.math.finite_differences.calculate(lambda diff_summed: self.f_calculate_with_diff([diff_squared, diff_summed]), diff_summed, f_x=f_p, bounds=None, accuracy_order=1)

        ## compose derivative
        d_f_p = np.matrix(d1_f_p) * np.matrix(d_diff_squared) + np.matrix(d2_f_p) * np.matrix(d_diff_summed)
        d_f_p = np.array(d_f_p.flat)

        return d_f_p



## Log normal distribution


class LWLS(BaseWeighted, BaseLog):

    def f_calculate(self, parameters):
        m = self.model_f(parameters)
        y = self.results
        v = self.variances

        # a = 2 * np.log(m) - 1/2 * np.log(m**2 + s**2)
        # b = np.log(m**2 + s**2) - 2 * np.log(m)
        c = v / m**2 + 1
        a = np.log(m / np.sqrt(c))
        b = np.log(c)

        r = a - np.log(y)

        f = np.sum(np.log(b) + r**2 / b)

        return f


    def df_calculate(self, parameters):
        m = self.model_f(parameters)
        dm = self.model_df(parameters)
        y = self.results
        v = self.variances

        a = 2 * np.log(m) - 1/2 * np.log(m**2 + v)
        da = m * (2/m**2 - 1/(m**2 + v))

        b = np.log(m**2 + v) - 2 * np.log(m)
        db = 2 * m * (1/(m**2 + v) - 1/m**2)

        r = a - np.log(y)

        # df_factor = (1 + 2*r - r**2/b) / b
        df_factor = (2*r*da + (1 - r**2/b)*db) / b
        df = np.sum(df_factor[:, np.newaxis] * dm, axis=0)

        return df



class LGLS(BaseGeneralized, BaseLog):

    def distribution_matrix(self, parameters):
        C = self.covariance_matrix
        F = self.model_f(parameters)
        F_MI = util.math.sparse.create.diag(1/F)
        C = F_MI * C * F_MI
        C.data = np.log(C.data + 1)
        return C

    def distribution_matrix_cholmod_factor_calculate(self, parameters):
        C = self.distribution_matrix(parameters)
        f = scikits.sparse.cholmod.cholesky(C)
        return f

    def distribution_matrix_cholmod_factor(self, parameters):
        return self.cache.memory_cache.get_value(parameters, 'distribution_matrix_cholmod_factor', self.distribution_matrix_cholmod_factor_calculate)





## Family


class Family(ndop.util.data_base.Family): 
   
    member_classes = {'WOA': [(OLS, [{}]), (WLS, [{}]), (LWLS, [{}])], 
                      'WOD': [(OLS, [{}]), (WLS, [{}]), (LWLS, [{}]), (GLS, [{'correlation_min_values': correlation_min_values, 'correlation_max_year_diff': float('inf')} for correlation_min_values in (40, 35, 30)])],
                      'WOD.1': [(OLS, [{}]), (WLS, [{}]), (LWLS, [{}]), (GLS, [{'correlation_min_values': correlation_min_values, 'correlation_max_year_diff': float('inf')} for correlation_min_values in (40, 35, 30, 25)])],
                      'WOD.0': [(OLS, [{}]), (WLS, [{}]), (LWLS, [{}]), (GLS, [{'correlation_min_values': correlation_min_values, 'correlation_max_year_diff': float('inf')} for correlation_min_values in (40, 35, 30, 25, 20)])]
                      } 

   
    # member_classes = {'WOA': [(OLS, [{}]), (WLS, [{}]), (LWLS, [{}])], 
    #                   'WOD': [(OLS, [{}]), (WLS, [{}]), (LWLS, [{}]), (GLS, [{'correlation_min_values': correlation_min_values, 'correlation_max_year_diff': float('inf')} for correlation_min_values in (40, 35, 30)])],
    #                   'WOD.1': [(GLS, [{'correlation_min_values': correlation_min_values, 'correlation_max_year_diff': float('inf')} for correlation_min_values in (25,)]), ],
    #                   'WOD.0': [(OLS, [{}]), (WLS, [{}]), (LWLS, [{}]), (GLS, [{'correlation_min_values': correlation_min_values, 'correlation_max_year_diff': float('inf')} for correlation_min_values in (40, 35, 30, 25, 20)])]
    #                   } 

    def f(self, parameters):
        fun = lambda o: o.f(parameters)
        value = self.get_function_value(fun)
        return value

    def f_normalized(self, parameters):
        fun = lambda o: o.f_normalized(parameters)
        value = self.get_function_value(fun)
        return value

    def df(self, parameters):
        fun = lambda o: o.df(parameters)
        value = self.get_function_value(fun)
        return value
