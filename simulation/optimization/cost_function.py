import os.path

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

import matrix.approximation.positive_semidefinite
import matrix.calculate

import simulation.model.constants
import simulation.model.options
import simulation.optimization.constants
import simulation.optimization.database
import simulation.util.cache

import measurements.all.data
import measurements.universal.data


# Base

class Base(simulation.util.cache.Cache):

    def __init__(self, measurements_object, model_options=None, model_job_options=None, include_initial_concentrations_factor_to_model_parameters=True, use_global_value_database=True):
        base_dir = simulation.optimization.constants.COST_FUNCTION_DIRNAME

        if model_job_options is None:
            model_job_options = {}
        try:
            model_job_options['name']
        except KeyError:
            self.measurements = measurements_object
            model_job_options['name'] = f'CF_{str(self)}'

        super().__init__(base_dir, measurements_object, model_options=model_options, model_job_options=model_job_options,
                         include_initial_concentrations_factor_to_model_parameters=include_initial_concentrations_factor_to_model_parameters)

        if use_global_value_database:
            self.global_value_database = simulation.optimization.database.database_for_cost_function(self)

    # model options
    @simulation.util.cache.Cache.model_options.setter
    def model_options(self, model_options):
        simulation.util.cache.Cache.model_options.fset(self, model_options)
        try:
            self.global_value_database
        except AttributeError:
            pass
        else:
            self.global_value_database = simulation.optimization.database.database_for_cost_function(self)

    # cost function values

    def _add_value_to_database(self, value, overwrite=False):
        try:
            db = self.global_value_database
        except AttributeError:
            pass
        else:
            concentrations = self.model.initial_constant_concentrations
            time_step = self.model_options.time_step
            parameters = self.model.parameters
            key = np.array([*concentrations, time_step, *parameters])
            db.set_value_with_key(key, value, use_tolerances=False, overwrite=overwrite)

    def normalize(self, value):
        return value / self.measurements.number_of_measurements

    def unnormalize(self, value):
        return value * self.measurements.number_of_measurements

    def f_calculate_unnormalized(self):
        raise NotImplementedError("Please implement this method.")

    def f_calculate_normalized(self):
        return self.normalize(self.f_calculate_unnormalized())

    def f(self, normalized=True):
        if normalized:
            value = self._value_from_file_cache(simulation.optimization.constants.COST_FUNCTION_F_FILENAME.format(normalized=True),
                                                self.f_calculate_normalized, derivative_used=False)
            self._add_value_to_database(value, overwrite=False)
        else:
            value = self.unnormalize(self.f(normalized=True))
        assert np.isfinite(value)
        return value

    def f_available(self, normalized=True):
        return self._value_in_file_cache(simulation.optimization.constants.COST_FUNCTION_F_FILENAME.format(normalized=True),
                                         derivative_used=False)

    def df_calculate_unnormalized(self):
        raise NotImplementedError("Please implement this method.")

    def df_calculate_normalized(self):
        return self.normalize(self.df_calculate_unnormalized())

    def df(self, normalized=True):
        # calculate df
        if normalized:
            def calculation_method():
                return self.df_calculate_normalized()
            file = simulation.optimization.constants.COST_FUNCTION_DF_FILENAME.format(
                normalized=normalized,
                include_total_concentration=self.include_initial_concentrations_factor_to_model_parameters)
            df = self._value_from_file_cache(file, calculation_method, derivative_used=True)
        else:
            df = self.unnormalize(self.df(normalized=True))

        # return
        assert df.shape[-1] == len(self.model_parameters)
        assert np.all(np.isfinite(df))
        return df

    def df_available(self, normalized=True):
        file = simulation.optimization.constants.COST_FUNCTION_DF_FILENAME.format(
            normalized=normalized,
            include_total_concentration=self.include_initial_concentrations_factor_to_model_parameters)
        return self._value_in_file_cache(file, derivative_used=True)


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

    def f_calculate_unnormalized(self):
        F = self.model_f()
        results = self.measurements_results()
        f = np.sum((F - results)**2)
        assert np.isfinite(f)
        return f

    def df_calculate_unnormalized(self):
        F = self.model_f()
        DF = self.model_df()
        results = self.measurements_results()
        residuals = F - results
        df = 2 * DF.T @ residuals
        assert np.all(np.isfinite(df))
        assert df.shape == (self.model_parameters_len,)
        return df


class WLS(BaseUsingStandardDeviation):

    def f_calculate_unnormalized(self):
        F = self.model_f()
        results = self.measurements_results()
        variances = self.measurements.variances

        f = np.sum((F - results)**2 / variances)
        assert np.isfinite(f)
        return f

    def df_calculate_unnormalized(self):
        F = self.model_f()
        DF = self.model_df(derivative_order=1)
        results = self.measurements_results()
        variances = self.measurements.variances
        weighted_residuals = (F - results) / variances
        df = 2 * DF.T @ weighted_residuals
        assert np.all(np.isfinite(df))
        assert df.shape == (self.model_parameters_len,)
        return df


class GLS(BaseUsingCorrelation):

    def f_calculate_unnormalized(self):
        F = self.model_f()
        results = self.measurements_results()
        standard_deviations = self.measurements.standard_deviations
        weighted_residual = (F - results) / standard_deviations
        correlation_matrix_decomposition = self.measurements.correlations_own_decomposition
        f = correlation_matrix_decomposition.inverse_matrix_both_sides_multiplication(weighted_residual)
        assert np.isfinite(f)
        return f

    def df_calculate_unnormalized(self, derivative_order=1):
        F = self.model_f()
        DF = self.model_df(derivative_order=1)
        results = self.measurements_results()
        standard_deviations = self.measurements.standard_deviations
        weighted_residuals = (F - results) / standard_deviations
        correlation_matrix_decomposition = self.measurements.correlations_own_decomposition
        factors = correlation_matrix_decomposition.inverse_matrix_right_side_multiplication(weighted_residual)
        factors = factors / standard_deviations
        df = 2 * DF.T @ factors
        assert np.all(np.isfinite(df))
        assert df.shape == (self.model_parameters_len,)
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

    def model_df(self):
        min_mask = super().model_f() < self.min_value
        df = super().model_df()
        df[min_mask] = 0
        return df

    def measurements_results(self):
        return np.maximum(super().measurements_results(), self.min_value)


class LWLS(BaseLog, BaseUsingStandardDeviation):

    @property
    def variances(self):
        return self.measurements.variances

    def distribution_parameter_my(self):
        expectations = self.model_f()
        variances = self.variances
        my = 2 * np.log(expectations) - 0.5 * np.log(expectations**2 + variances)
        return my

    def df_distribution_parameter_my(self):
        expectations = self.model_f()
        df_expectations = self.model_df()
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

    def df_distribution_parameter_sigma_diagonal(self):
        expectations = self.model_f()
        df_expectations = self.model_df()
        df_factor = -2 * expectations / (expectations**2 + 1)
        df_sigma_diagonal = df_factor[:, np.newaxis] * df_expectations
        return df_sigma_diagonal

    def f_calculate_unnormalized(self):
        results = self.measurements_results()
        my = self.distribution_parameter_my()
        sigma_diagonal = self.distribution_parameter_sigma_diagonal()
        f = np.sum(np.log(sigma_diagonal))
        f += np.sum((np.log(results) - my)**2 / sigma_diagonal)
        return f

    def df_calculate_unnormalized(self):
        results = self.measurements_results()
        my = self.distribution_parameter_my()
        sigma_diagonal = self.distribution_parameter_sigma_diagonal()
        df_my = self.df_distribution_parameter_my()
        df_sigma_diagonal = self.df_distribution_parameter_sigma_diagonal()
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
        correlation_matrix_decomposition = matrix.approximation.positive_semidefinite.decomposition(
            correlation_matrix, min_diag_B=1, max_diag_B=1,
            min_diag_D=self.measurements.min_diag_value_decomposition_correlation,
            permutation=self.measurements.permutation_method_decomposition_correlation,
            return_type=matrix.constants.LDL_DECOMPOSITION_TYPE, overwrite_A=True)
        sigma = correlation_matrix_decomposition.inverse_matrix_both_sides_multiplication(weighted_residual, dtype=np.float128)
        sigma.data = np.log(sigma.data + 1)
        return sigma

    def distribution_parameter_sigma_decomposition(self):
        decomposition = matrix.calculate.decompose(self.distribution_parameter_sigma(), permutation=self.measurements.permutation_method_decomposition_correlation, check_finite=False, return_type=matrix.constants.LDL_DECOMPOSITION_TYPE)
        return decomposition

    def f_calculate_unnormalized(self):
        results = self.measurements_results()
        my = self.distribution_parameter_my()
        sigma_decomposition = self.distribution_parameter_sigma_decomposition()
        diff = np.log(results) - my
        f = sigma_decomposition.inverse_matrix_both_sides_multiplication(diff)
        f += np.sum(np.log(sigma_decomposition.d))
        return f


# classes lists

ALL_COST_FUNCTION_CLASSES_WITHOUT_STANDARD_DEVIATION = [OLS, ]
ALL_COST_FUNCTION_CLASSES_ONLY_WITH_STANDARD_DEVIATION = [WLS, LOLS, LWLS]
ALL_COST_FUNCTION_CLASSES_WITH_CORRELATION = [GLS, LGLS]
ALL_COST_FUNCTION_CLASSES = ALL_COST_FUNCTION_CLASSES_WITHOUT_STANDARD_DEVIATION + ALL_COST_FUNCTION_CLASSES_ONLY_WITH_STANDARD_DEVIATION + ALL_COST_FUNCTION_CLASSES_WITH_CORRELATION

ALL_COST_FUNCTION_NAMES = [cost_function_class.__name__ for cost_function_class in ALL_COST_FUNCTION_CLASSES]


# iterator

def cost_functions_for_all_measurements(min_measurements_standard_deviations=None, min_measurements_correlations=None,
                                        min_standard_deviations=None, min_diag_correlations=None,
                                        max_box_distance_to_water=None, cost_function_classes=None, model_options=None):
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
        min_measurements_standard_deviation=min_measurements_standard_deviations,
        min_measurements_correlation=min_measurements_correlations,
        min_standard_deviation=min_standard_deviations,
        min_diag_correlations=min_diag_correlations,
        max_box_distance_to_water=max_box_distance_to_water,
        water_lsm='TMM',
        sample_lsm='TMM')

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


def iterator(cost_functions, model_names=None, time_steps=None):
    if cost_functions is None:
        cost_functions = []

    if len(cost_functions) > 0:
        # default values
        if model_names is None:
            model_names = simulation.model.constants.MODEL_NAMES
        if time_steps is None:
            time_steps = simulation.model.constants.METOS_TIME_STEPS

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
            for model_options in model.iterator(model_names=[model_name], time_steps=time_steps):
                for cost_function in cost_functions:
                    cost_function.model_options = model_options
                    yield cost_function

        # reset to original model and measurements
        for cost_function, original_model, original_measurements in zip(cost_functions, original_model_list, original_measurements_list):
            cost_function.model = original_model
            cost_function.measurements = original_measurements
