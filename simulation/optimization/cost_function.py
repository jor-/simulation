import os.path
import warnings

import numpy as np

import simulation.model.cache
import simulation.model.constants
import simulation.model.options
import simulation.optimization.constants

import measurements.universal.data

import util.math.optimize.with_scipy
import util.math.finite_differences
import util.math.sparse.create
import util.math.sparse.solve
from util.math.matrix import SingularMatrixError

import util.logging
logger = util.logging.logger



## Base

class Base():

    def __init__(self, measurements_collection, model_options=None, job_options=None):
        ## set measurements
        self.measurements = measurements.universal.data.as_measurements_collection(measurements_collection)
        
        ## prepare job options
        if job_options is None:
            job_options = {}
        try:
            job_options['name']
        except KeyError:
            job_options['name'] = str(self)

        try:
            job_options['nodes_setup']
        except KeyError:
            try:
                job_options['spinup']
            except KeyError:
                job_options['spinup'] = {}
            try:
                job_options['spinup']['nodes_setup']
            except KeyError:
                job_options['spinup']['nodes_setup'] = simulation.optimization.constants.COST_FUNCTION_NODES_SETUP_SPINUP.copy()
            try:
                job_options['derivative']
            except KeyError:
                job_options['derivative'] = {}
            try:
                job_options['derivative']['nodes_setup']
            except KeyError:
                job_options['derivative']['nodes_setup'] = simulation.optimization.constants.COST_FUNCTION_NODES_SETUP_DERIVATIVE.copy()
            try:
                job_options['trajectory']
            except KeyError:
                job_options['trajectory'] = {}
            try:
                job_options['trajectory']['nodes_setup']
            except KeyError:
                job_options['trajectory']['nodes_setup'] = simulation.optimization.constants.COST_FUNCTION_NODES_SETUP_TRAJECTORY.copy()

        ## set model, initial_base_concentrations and cache
        self.model = simulation.model.cache.Model(model_options=model_options, job_options=job_options)
        self.initial_base_concentrations = np.asanyarray(self.model.model_options.initial_concentration_options.concentrations)
        self.cache = self.model._cache


    @property
    def measurements(self):
        return self._measurements
    
    @measurements.setter
    def measurements(self, measurements_collection):
        self._measurements = measurements.universal.data.as_measurements_collection(measurements_collection)
    
    
    @property
    def parameters(self):
        return self._parameters
    
    
    @parameters.setter
    def parameters(self, parameters):
        parameters = np.asanyarray(parameters)
        
        model_parameters_len = self.model.model_options.parameters_len
        if len(parameters) == model_parameters_len:
            self.model.model_options.parameters = parameters
        elif len(parameters) == model_parameters_len + 1:
            self.model.model_options.parameters = parameters[:-1]
            self.model.model_options.initial_concentration_options.concentrations = self.initial_base_concentrations * parameters[-1]
        else:
            raise ValueError('The parameters for the model {} must be a vector of length {} or {}, but its length is {}.'.format(self.model.model_options.model_name, model_parameters_len, model_parameters_len + 1, len(parameters)))
        
        self._parameters = parameters
    
    
    @property
    def parameters_include_initial_concentrations_factor(self):        
        return len(self.parameters) == self.model.model_options.parameters_len + 1

    
    @property
    def name(self):
        return self.__class__.__name__
    
    
    @property
    def _measurements_name(self):
        return str(self.measurements)
        

    def __str__(self):
        return '{}({})'.format(self.name, self._measurements_name)


    @property
    def _cache_dirname(self):
        return os.path.join(simulation.optimization.constants.COST_FUNCTION_DIRNAME, self._measurements_name, self.name)
    
    
    def _filename(self, filename):
        return os.path.join(self._cache_dirname, filename)


    ## cost function values

    def f_calculate(self):
        raise NotImplementedError("Please implement this method.")

    def f(self):
        filename = self._filename(simulation.optimization.constants.COST_FUNCTION_F_FILENAME)
        return self.cache.get_value(filename, self.f_calculate, derivative_used=False, save_also_txt=True)


    def f_available(self):
        filename = self._filename(simulation.optimization.constants.COST_FUNCTION_F_FILENAME)
        return self.cache.has_value(filename)
    

    def f_normalized_calculate(self):
        f = self.f()
        m = self.measurements.number_of_measurements
        f_normalized = f / m
        return f_normalized

    def f_normalized(self):
        filename = self._filename(simulation.optimization.constants.COST_FUNCTION_F_NORMALIZED_FILENAME)
        return self.cache.get_value(filename, self.f_normalized_calculate, derivative_used=False, save_also_txt=True)


    def df_calculate(self, derivative_kind):
        raise NotImplementedError("Please implement this method.")

    def df(self):
        ## get needed derivative kinds
        derivative_kinds = ['model_parameters']
        if self.parameters_include_initial_concentrations_factor:
            derivative_kinds.append('total_concentration_factor')

        filename_pattern = self._filename(simulation.optimization.constants.COST_FUNCTION_DF_FILENAME.format(step_size=self.model.model_options.derivative_options.step_size, derivative_kind='{derivative_kind}'))
        
        ## calculate and cache derivative for each kind
        df = []
        for derivative_kind in derivative_kinds:
            filename = filename_pattern.format(derivative_kind=derivative_kind)
            df_i = self.cache.get_value(filename, lambda: self.df_calculate(derivative_kind), derivative_used=True, save_also_txt=True)
            df.append(df_i)
        
        ## concatenate to one df
        df = np.concatenate(df, axis=-1)
        
        ## return
        assert df.shape[-1] == len(self.parameters)
        return df


    def df_available(self):
        ## get needed derivative kinds
        derivative_kinds = ['model_parameters']
        if self.parameters_include_initial_concentrations_factor:
            derivative_kinds.append('total_concentration_factor')

        filename_pattern = self._filename(simulation.optimization.constants.COST_FUNCTION_DF_FILENAME.format(step_size=self.model.model_options.derivative_options.step_size, derivative_kind='{derivative_kind}'))
        
        ## check cache derivative for each kind
        return all(self.cache.has_value(filename_pattern.format(derivative_kind=derivative_kind)) for derivative_kind in derivative_kinds)


    ## model and data values
    
    def model_f(self):
        f = self.model.f_measurements(*self.measurements)
        f = self.measurements.convert_measurements_dict_to_array(f)
        assert len(f) == self.measurements.number_of_measurements
        return f
    
    def model_df(self, derivative_kind):
        df = self.model.df_measurements(*self.measurements, partial_derivative_kind=derivative_kind)
        df = self.measurements.convert_measurements_dict_to_array(df)
        assert len(df) == self.measurements.number_of_measurements
        return df
    
    def results(self):
        results = self.measurements.values
        assert len(results) == self.measurements.number_of_measurements
        return results



## Normal distribution

class OLS(Base):

    def f_calculate(self):
        F = self.model_f()
        results = self.results()

        f = np.sum((F - results)**2)

        return f


    def f_normalized_calculate(self):
        f_normalized = super().f_normalized_calculate()
        inverse_average_variance = 1 / ((self.measurements.variances).mean())
        f_normalized = f_normalized * inverse_average_variance
        return f_normalized


    def df_calculate(self, derivative_kind):
        F = self.model_f()
        DF = self.model_df(derivative_kind)
        results = self.results()

        df_factors = F - results
        df = 2 * np.sum(df_factors[:, np.newaxis] * DF, axis=0)

        return df


class WLS(Base):

    def f_calculate(self):
        F = self.model_f()
        results = self.results()
        inverse_variances = 1 / self.measurements.variances

        f = np.sum((F - results)**2 * inverse_variances)

        return f


    def df_calculate(self, derivative_kind):
        F = self.model_f()
        DF = self.model_df(derivative_kind)
        results = self.results()
        inverse_variances = 1 / self.measurements.variances

        df_factors = (F - results) * inverse_variances
        df = 2 * np.sum(df_factors[:, np.newaxis] * DF, axis=0)

        return df


class GLS(Base):

    def f_calculate(self):
        F = self.model_f()
        results = self.results()
        inverse_deviations = 1 / self.measurements.standard_deviations
        correlation_matrix_cholesky_decomposition = self.measurements.correlations_own_cholesky_decomposition
        P = correlation_matrix_cholesky_decomposition['P']
        L = correlation_matrix_cholesky_decomposition['L']
        
        weighted_residual =  (F - results) * inverse_deviations
        inv_L_mul_weighted_residual = util.math.sparse.solve.forward_substitution(L, P * weighted_residual)
        
        f = np.sum(inv_L_mul_weighted_residual**2)
        return f


    def df_calculate(self, derivative_kind):
        DF = self.model_df(derivative_kind)
        inverse_deviations = 1 / self.measurements.standard_deviations
        correlation_matrix_cholesky_decomposition = self.measurements.correlations_own_cholesky_decomposition
        P = correlation_matrix_cholesky_decomposition['P']
        L = correlation_matrix_cholesky_decomposition['L']

        weighted_residual =  (F - results) * inverse_deviations
        inv_L_mul_weighted_residual = util.math.sparse.solve.forward_substitution(L, P * weighted_residual)
        
        inv_C_mul_weighted_residual = util.math.sparse.solve.backward_substitution(L.T, inv_L_mul_weighted_residual)
        inv_C_mul_weighted_residual = P.T * inv_C_mul_weighted_residual

        df_factors = inv_C_mul_weighted_residual * inverse_deviations

        df = 2 * np.sum(df_factors[:,np.newaxis] * DF, axis=0)
        return df



## Log normal distribution

class BaseLog(Base):

    def __init__(self, *args, **kargs):
        from .constants import CONCENTRATION_MIN_VALUE
        self.min_value = CONCENTRATION_MIN_VALUE

        super().__init__(*args, **kargs)

    def model_f(self):
        return np.maximum(super().model_f(), self.min_value)

    def model_df(self, derivative_kind):
        min_mask = super().model_f() < self.min_value
        df = super().model_df(derivative_kind)
        df[min_mask] = 0
        return df

    def results(self):
        return np.maximum(super().results(), self.min_value)



class LWLS(BaseLog):

    def f_calculate(self):
        m = self.model_f()
        y = self.results()
        v = self.measurements.variances

        c = v / m**2 + 1
        a = np.log(m / np.sqrt(c))
        b = np.log(c)

        r = a - np.log(y)

        f = np.sum(np.log(b) + r**2 / b)

        return f


    def df_calculate(self, derivative_kind):
        m = self.model_f()
        dm = self.model_df(derivative_kind)
        y = self.results()
        v = self.measurements.variances

        a = 2 * np.log(m) - 1/2 * np.log(m**2 + v)
        da = m * (2/m**2 - 1/(m**2 + v))

        b = np.log(m**2 + v) - 2 * np.log(m)
        db = 2 * m * (1/(m**2 + v) - 1/m**2)

        r = a - np.log(y)

        df_factor = (2*r*da + (1 - r**2/b)*db) / b
        df = np.sum(df_factor[:, np.newaxis] * dm, axis=0)

        return df



class LGLS(BaseLog):

    def distribution_matrix(self):
        D = util.math.sparse.create.diag(self.measurements.standard_deviations)
        C = D * self.measurements.correlations() * D
        F = self.model_f()
        F_MI = util.math.sparse.create.diag(1/F)
        C = F_MI * C * F_MI
        C.data = np.log(C.data + 1)
        return C

    def distribution_matrix_cholmod_factor(self):
        import util.math.sparse.decompose.with_cholmod
        C = self.distribution_matrix()
        f = util.math.sparse.decompose.with_cholmod.cholmod.cholesky(C)
        return f



## class lists

ALL_COST_FUNCTION_CLASSES_WITHOUT_CORRELATION = [OLS, WLS, LWLS]
ALL_COST_FUNCTION_CLASSES_WITH_CORRELATION = [GLS, LGLS]
ALL_COST_FUNCTION_CLASSES = ALL_COST_FUNCTION_CLASSES_WITHOUT_CORRELATION + ALL_COST_FUNCTION_CLASSES_WITH_CORRELATION



## iterator

def cost_functions_for_all_measurements(max_box_distance_to_water_list=None, min_measurements_correlations_list=None, cost_function_classes=None, model_options=None):
    ## default values
    if max_box_distance_to_water_list is None:
        max_box_distance_to_water_list = [0, 1, float('inf')]
    if min_measurements_correlations_list is None:
        min_measurements_correlations_list = [float('inf')]
    if cost_function_classes is None:
        cost_function_classes = ALL_COST_FUNCTION_CLASSES    
    if model_options is None:
        model_options = simulation.model.options.ModelOptions()
        model_options.spinup_options = {'years':1, 'tolerance':0.0, 'combination':'or'}
    
    ## split cost function classes
    cost_function_classes = set(cost_function_classes)
    cost_function_classes_without_correlation = cost_function_classes & set(ALL_COST_FUNCTION_CLASSES_WITHOUT_CORRELATION)
    cost_function_classes_with_correlation = cost_function_classes & set(ALL_COST_FUNCTION_CLASSES_WITH_CORRELATION)

    ## init all cost functions
    cost_functions = []
    for max_box_distance_to_water in max_box_distance_to_water_list:
        for i in range(min_measurements_correlations_list):
            min_measurements_correlations = min_measurements_correlations_list[i]
            
            measurements_collection = measurements.all.pw.data.all_measurements(max_box_distance_to_water=max_box_distance_to_water, min_measurements_correlations=min_measurements_correlations)
            
            if len(cost_function_classes_without_correlation) > 0 and i == 0:
                cost_functions.extend([cost_functions_class(measurements_collection) for cost_functions_class in cost_function_classes_without_correlation])

            if len(cost_function_classes_with_correlation) > 0 and min_measurements_correlations != float('inf'):
                cost_functions.extend([cost_functions_class(measurements_collection) for cost_functions_class in cost_function_classes_with_correlation])
    
    ## set same model and model options
    if len(cost_functions) > 0:
        model = cost_functions[0].model
        model.model_options = model_options
        for cost_function in cost_functions:
            cost_function.model = model



def iterator(cost_functions, model_names=None):
    ## default values
    if model_names is None:
        model_names = simulation.model.constants.MODEL_NAMES

    ## set same model and model options, store original measurements
    model = cost_functions[0].model
    model.model_options = model_options
    original_measurements_list = []
    for cost_function in cost_functions:
        cost_function.model = model
        original_measurements.append(cost_function.measurements)
    
    ## iterate over models
    for model_name in model_names:
        ## set model name
        model_options.model_name = model_name
        ## set measurements
        for cost_function, original_measurements in zip(cost_functions, original_measurements_list):
            measurements_for_model = original_measurements.subset(model_options.tracers)
            cost_function.measurements = measurements_for_model
        ## iterate over other options
        for model_options in model.iterator(model_names=[model_name]):
            for cost_function in cost_functions:
                yield cost_function
