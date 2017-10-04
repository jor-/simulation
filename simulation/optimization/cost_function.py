import os.path

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matrix.calculate

import simulation.model.cache
import simulation.model.constants
import simulation.model.options
import simulation.optimization.constants

import measurements.all.data
import measurements.universal.data


# Base

class Base():

    def __init__(self, measurements_object, model_options=None, model_job_options=None):
        # set measurements,
        self.measurements = measurements_object

        # prepare job options
        if model_job_options is None:
            model_job_options = {}
        try:
            model_job_options['name']
        except KeyError:
            model_job_options['name'] = str(self)

        try:
            model_job_options['nodes_setup']
        except KeyError:
            try:
                model_job_options['spinup']
            except KeyError:
                model_job_options['spinup'] = {}
            try:
                model_job_options['spinup']['nodes_setup']
            except KeyError:
                try:
                    model_job_options['spinup']['nodes_setup'] = simulation.optimization.constants.COST_FUNCTION_NODES_SETUP_SPINUP.copy()
                except AttributeError:
                    pass
            try:
                model_job_options['derivative']
            except KeyError:
                model_job_options['derivative'] = {}
            try:
                model_job_options['derivative']['nodes_setup']
            except KeyError:
                try:
                    model_job_options['derivative']['nodes_setup'] = simulation.optimization.constants.COST_FUNCTION_NODES_SETUP_DERIVATIVE.copy()
                except AttributeError:
                    pass
            try:
                model_job_options['trajectory']
            except KeyError:
                model_job_options['trajectory'] = {}
            try:
                model_job_options['trajectory']['nodes_setup']
            except KeyError:
                try:
                    model_job_options['trajectory']['nodes_setup'] = simulation.optimization.constants.COST_FUNCTION_NODES_SETUP_TRAJECTORY.copy()
                except AttributeError:
                    pass

        # set model and initial_base_concentrations
        self.model = simulation.model.cache.Model(model_options=model_options, job_options=model_job_options)
        self.initial_base_concentrations = np.asanyarray(self.model.model_options.initial_concentration_options.concentrations)

    # cache, measurements, parameters

    @property
    def cache(self):
        return self.model._cache

    @property
    def measurements(self):
        return self._measurements

    @measurements.setter
    def measurements(self, measurements_object):
        measurements_object = measurements.universal.data.as_measurements_collection(measurements_object)
        self._measurements = measurements_object

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

    # names

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

    # cost function values

    def f_calculate(self):
        raise NotImplementedError("Please implement this method.")

    def f(self):
        filename = self._filename(simulation.optimization.constants.COST_FUNCTION_F_FILENAME)
        return self.cache.get_value(filename, self.f_calculate, derivative_used=False, save_also_txt=True)

    def f_available(self):
        filename = self._filename(simulation.optimization.constants.COST_FUNCTION_F_FILENAME)
        return self.cache.has_value(filename, derivative_used=False)

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
        # get needed derivative kinds
        derivative_kinds = ['model_parameters']
        if self.parameters_include_initial_concentrations_factor:
            derivative_kinds.append('total_concentration_factor')

        filename_pattern = self._filename(simulation.optimization.constants.COST_FUNCTION_DF_FILENAME.format(derivative_kind='{derivative_kind}'))

        # calculate and cache derivative for each kind
        df = []
        for derivative_kind in derivative_kinds:
            filename = filename_pattern.format(derivative_kind=derivative_kind)
            df_i = self.cache.get_value(filename, lambda: self.df_calculate(derivative_kind), derivative_used=True, save_also_txt=True)
            df.append(df_i)

        # concatenate to one df
        df = np.concatenate(df, axis=-1)

        # return
        assert df.shape[-1] == len(self.parameters)
        return df

    def df_available(self):
        # get needed derivative kinds
        derivative_kinds = ['model_parameters']
        if self.parameters_include_initial_concentrations_factor:
            derivative_kinds.append('total_concentration_factor')

        filename_pattern = self._filename(simulation.optimization.constants.COST_FUNCTION_DF_FILENAME.format(derivative_kind='{derivative_kind}'))

        # check cache derivative for each kind
        return all(self.cache.has_value(filename_pattern.format(derivative_kind=derivative_kind), derivative_used=True) for derivative_kind in derivative_kinds)

    # model and data values

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


class BaseUsingStandardDeviation(Base):

    @property
    def name(self):
        name = super().name
        standard_deviation_id = self.measurements.standard_deviation_id
        if len(standard_deviation_id) > 0:
            name = name + '(' + standard_deviation_id + ')'
        return name


class BaseUsingCorrelation(Base):

    @property
    def name(self):
        name = super().name
        correlation_id = self.measurements.correlation_id
        if len(correlation_id) > 0:
            name = name + '(' + correlation_id + ')'
        return name


# Normal distribution

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


class WLS(BaseUsingStandardDeviation):

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


class GLS(BaseUsingCorrelation):

    def f_calculate(self):
        F = self.model_f()
        results = self.results()
        inverse_deviations = 1 / self.measurements.standard_deviations
        weighted_residual = (F - results) * inverse_deviations
        correlation_matrix_decomposition = self.measurements.correlations_own_decomposition
        f = correlation_matrix_decomposition.inverse_matrix_both_sides_multiplication(weighted_residual)
        return f

    def df_calculate(self, derivative_kind):
        F = self.model_f()
        results = self.results()
        inverse_deviations = 1 / self.measurements.standard_deviations
        weighted_residual = (F - results) * inverse_deviations
        DF = self.model_df(derivative_kind)
        correlation_matrix_decomposition = self.measurements.correlations_own_decomposition
        inverse_correlation_matrix_right_side_multiplied_weighted_residual = correlation_matrix_decomposition.inverse_matrix_right_side_multiplication(weighted_residual)
        df_factors = inverse_correlation_matrix_right_side_multiplied_weighted_residual * inverse_deviations

        df = 2 * np.sum(df_factors[:, np.newaxis] * DF, axis=0)
        return df


# Log normal distribution

class BaseLog(Base):

    def __init__(self, *args, **kargs):
        from .constants import CONCENTRATION_MIN_VALUE
        self.min_value = CONCENTRATION_MIN_VALUE

        super().__init__(*args, **kargs)

    @property
    def name(self):
        return '{name}_(min_{min_value})'.format(name=super().name, min_value=self.min_value)

    def model_f(self):
        return np.maximum(super().model_f(), self.min_value)

    def model_df(self, derivative_kind):
        min_mask = super().model_f() < self.min_value
        df = super().model_df(derivative_kind)
        df[min_mask] = 0
        return df

    def results(self):
        return np.maximum(super().results(), self.min_value)


class LWLS(BaseLog, BaseUsingStandardDeviation):

    @property
    def variances(self):
        return self.measurements.variances

    def distribution_parameter_my(self):
        expectations = self.model_f()
        variances = self.variances
        my = 2 * np.log(expectations) - 0.5 * np.log(expectations**2 + variances)
        return my

    def df_distribution_parameter_my(self, derivative_kind):
        expectations = self.model_f()
        df_expectations = self.model_df(derivative_kind)
        variances = self.variances

        expectations_squared = expectations**2
        df_factor = (2 / expectations) - (expectations / (expectations_squared + variances))
        df_my = df_factor[:, np.newaxis] * df_expectations
        return df_my

    def distribution_parameter_sigma_diagonal(self):
        expectations = self.model_f()
        variances = self.variances
        sigma_diagonal = np.log(variances / expectations**2 + 1)
        return sigma_diagonal

    def df_distribution_parameter_sigma_diagonal(self, derivative_kind):
        expectations = self.model_f()
        df_expectations = self.model_df(derivative_kind)

        df_factor = -2 * expectations / (expectations**2 + 1)
        df_sigma_diagonal = df_factor[:, np.newaxis] * df_expectations
        return df_sigma_diagonal

    def f_calculate(self):
        results = self.results()
        my = self.distribution_parameter_my()
        sigma_diagonal = self.distribution_parameter_sigma_diagonal()

        f = np.sum(np.log(sigma_diagonal))
        f += np.sum((np.log(results) - my)**2 / sigma_diagonal)

        return f

    def df_calculate(self, derivative_kind):
        results = self.results()
        my = self.distribution_parameter_my()
        sigma_diagonal = self.distribution_parameter_sigma_diagonal()
        df_my = self.df_distribution_parameter_my(derivative_kind)
        df_sigma_diagonal = self.df_distribution_parameter_sigma_diagonal(derivative_kind)

        df = np.sum((1 / sigma_diagonal)[:, np.newaxis] * df_sigma_diagonal, axis=0)

        df_factor = (my - np.log(results)) / sigma_diagonal
        df += np.sum(df_factor[:, np.newaxis] * (2 * df_my - df_factor[:, np.newaxis] * df_sigma_diagonal), axis=0)
        return df


class LOLS(LWLS):

    @property
    def variances(self):
        return self.measurements.variances.mean()


class LGLS(BaseUsingCorrelation, BaseLog):

    def distribution_parameter_my(self):
        expectations = self.model_f()
        variances = self.measurements.variances
        my = 2 * np.log(expectations) - 0.5 * np.log(expectations**2 + variances)
        return my

    def distribution_parameter_sigma(self):
        expectations = self.model_f()
        standard_deviations_diag_matrix = scipy.sparse.diags(self.measurements.standard_deviations)
        correlation_matrix = self.measurements.correlations()

        correlation_matrix.data[correlation_matrix.data < 0] = 0        # set negative correlations to zero (since it must hold C_ij >= - E_i E_j)
        correlation_matrix.eliminate_zeros()
        correlation_matrix = matrix.calculate.approximate_decomposition(correlation_matrix, min_diag_value=self.measurements.min_diag_value_decomposition_correlation, min_abs_value=self.measurements.min_abs_correlation, permutation_method=self.measurements.permutation_method_decomposition_correlation, check_finite=False, return_type=matrix.constants.LDL_DECOMPOSITION_TYPE, overwrite_A=True)

        covariance_matrix = standard_deviations_diag_matrix * correlation_matrix * standard_deviations_diag_matrix

        inverse_expectations_diag_matrix = scipy.sparse.diags(1 / expectations)
        sigma = inverse_expectations_diag_matrix * covariance_matrix * inverse_expectations_diag_matrix
        sigma.data = np.log(sigma.data + 1)
        return sigma

    def distribution_parameter_sigma_decomposition(self):
        decomposition = matrix.decompose(self.distribution_parameter_sigma(), permutation_method=self.measurements.permutation_method_decomposition_correlation, check_finite=False, return_type=matrix.constants.LDL_DECOMPOSITION_TYPE)
        return decomposition

    def f_calculate(self):
        results = self.results()
        my = self.distribution_parameter_my()
        sigma_decomposition = self.distribution_parameter_sigma_decomposition()
        diff = np.log(results) - my
        f = sigma_decomposition.inverse_matrix_both_sides_multiplication(diff)
        f += np.sum(np.log(sigma_decomposition.d))
        return f


# class lists

ALL_COST_FUNCTION_CLASSES_WITHOUT_STANDARD_DEVIATION = [OLS, ]
ALL_COST_FUNCTION_CLASSES_ONLY_WITH_STANDARD_DEVIATION = [WLS, LOLS, LWLS]
ALL_COST_FUNCTION_CLASSES_WITH_CORRELATION = [GLS, LGLS]
ALL_COST_FUNCTION_CLASSES = ALL_COST_FUNCTION_CLASSES_WITHOUT_STANDARD_DEVIATION + ALL_COST_FUNCTION_CLASSES_ONLY_WITH_STANDARD_DEVIATION + ALL_COST_FUNCTION_CLASSES_WITH_CORRELATION


# iterator

def cost_functions_for_all_measurements(min_standard_deviations=None, min_measurements_correlations=None, max_box_distance_to_water=None, cost_function_classes=None, model_options=None):
    # default values
    if cost_function_classes is None:
        cost_function_classes = ALL_COST_FUNCTION_CLASSES
    if model_options is None:
        model_options = simulation.model.options.ModelOptions()

    # split cost function classes
    cost_function_classes = set(cost_function_classes)
    cost_function_classes_without_standard_deviation = cost_function_classes & set(ALL_COST_FUNCTION_CLASSES_WITHOUT_STANDARD_DEVIATION)
    cost_function_classes_only_with_standard_deviation = cost_function_classes & set(ALL_COST_FUNCTION_CLASSES_ONLY_WITH_STANDARD_DEVIATION)
    cost_function_classes_with_correlation = cost_function_classes & set(ALL_COST_FUNCTION_CLASSES_WITH_CORRELATION)

    # init all cost functions
    cost_functions = []
    measurements_collection = measurements.all.data.all_measurements(
        tracers=model_options.tracers,
        min_standard_deviations=min_standard_deviations,
        min_measurements_correlations=min_measurements_correlations,
        max_box_distance_to_water=max_box_distance_to_water)

    if len(cost_function_classes_without_standard_deviation) > 0:
        cost_functions.extend([cost_functions_class(measurements_collection) for cost_functions_class in cost_function_classes_without_standard_deviation])

    if len(cost_function_classes_only_with_standard_deviation) > 0:
        cost_functions.extend([cost_functions_class(measurements_collection) for cost_functions_class in cost_function_classes_only_with_standard_deviation])

    if len(cost_function_classes_with_correlation) > 0 and min_measurements_correlations != float('inf'):
        cost_functions.extend([cost_functions_class(measurements_collection) for cost_functions_class in cost_function_classes_with_correlation])

    # set same model and model options
    if len(cost_functions) > 0:
        model = cost_functions[0].model
        model.model_options = model_options
        for cost_function in cost_functions:
            cost_function.model = model

    return cost_functions


def iterator(cost_functions, model_names=None):
    if cost_functions is None:
        cost_functions = []

    if len(cost_functions) > 0:
        # default values
        if model_names is None:
            model_names = simulation.model.constants.MODEL_NAMES

        # set same model and model options, store original model and measurements
        model = cost_functions[0].model
        model_options = model.model_options
        original_model_list = []
        original_measurements_list = []
        for cost_function in cost_functions:
            original_model_list.append(cost_function.model)
            original_measurements_list.append(cost_function.measurements)
            cost_function.model = model

        # iterate over models
        for model_name in model_names:
            # set model name
            model_options.model_name = model_name
            # set measurements
            for cost_function, original_measurements in zip(cost_functions, original_measurements_list):
                measurements_for_model = original_measurements.subset(model_options.tracers)
                cost_function.measurements = measurements_for_model
            # iterate over other options
            for model_options in model.iterator(model_names=[model_name]):
                for cost_function in cost_functions:
                    yield cost_function

        # reset to original model and measurements
        for cost_function, original_model, original_measurements in zip(cost_functions, original_model_list, original_measurements_list):
            cost_function.model = original_model
            cost_function.measurements = original_measurements
