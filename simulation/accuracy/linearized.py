import math

import numpy as np
import scipy.linalg
import scipy.stats

import matrix

import util.logging
import util.math.util
import util.parallel.with_multiprocessing

import measurements.plot.util

import simulation.accuracy.constants
import simulation.optimization.cost_function
import simulation.util.cache


class Base(simulation.util.cache.Cache):

    def __init__(self, measurements_object, model_options=None, model_job_options=None):
        base_dir = simulation.accuracy.constants.CACHE_DIRNAME

        if model_job_options is None:
            model_job_options = {}
        try:
            model_job_options['name']
        except KeyError:
            self.measurements = measurements_object
            model_job_options['name'] = f'A_{str(self)}'

        super().__init__(base_dir, measurements_object, model_options=model_options, model_job_options=model_job_options,
                         include_initial_concentrations_factor_to_model_parameters=True)

    # *** uncertainty factors *** #

    def confidence_factor(self, alpha, include_variance_factor=True):
        assert 0 < alpha < 1
        if include_variance_factor:
            alpha = 1 - ((1 - alpha) / (2 * self.model_parameters_len))
            gamma = (1 + alpha) / 2
            return scipy.stats.t.ppf(gamma, self.measurements.number_of_measurements - self.model_parameters_len)
        else:
            gamma = (1 + alpha) / 2
            return scipy.stats.norm.ppf(gamma)

    @property
    def _cost_function_class(self):
        raise NotImplementedError("Please implement this method")

    @property
    def _cost_function(self):
        cf = self._cost_function_class(
            self.measurements, model_options=self.model.model_options, model_job_options=self.model.job_options,
            include_initial_concentrations_factor_to_model_parameters=self.include_initial_concentrations_factor_to_model_parameters)
        return cf

    @property
    def variance_factor(self):
        cf = self._cost_function
        f = cf.f(normalized=False)
        factor = f / self.measurements.number_of_measurements
        return factor

    # *** uncertainty model parameters *** #

    def information_matrix_type_F(self):
        raise NotImplementedError("Please implement this method")

    def information_matrix_type_F_with_additional_increase_only(self, **kwargs):
        raise NotImplementedError("Please implement this method")

    def information_matrix_type_F_with_additional_independent_increase_only(self, df_additional, standard_deviations_additional):
        correlation_matrix = scipy.sparse.eye(len(standard_deviations_additional))
        model_parameter_information_matrix = self.information_matrix_type_F_with_additional_increase_only(df=df_additional, standard_deviations=standard_deviations_additional, correlation_matrix=correlation_matrix)
        return model_parameter_information_matrix

    def information_matrix_type_F_with_additional_independent(self, df_additional, standard_deviations_additional):
        matrix = self.information_matrix(matrix_type='F')
        increase = self.information_matrix_type_F_with_additional_independent_increase_only(df_additional, standard_deviations_additional)
        return matrix + increase

    def information_matrix_type_H(self):
        cf = self._cost_function
        H = cf.df(derivative_order=2, normalized=False) * 0.5
        return H

    def information_matrix(self, matrix_type='F'):
        util.logging.debug(f'Calculating model parameter information matrix of type {matrix_type}.')

        filename = simulation.accuracy.constants.INFORMATION_MATRIX_FILENAME.format(matrix_type=matrix_type)
        if matrix_type == 'F':
            calculation_method = self.information_matrix_type_F
        elif matrix_type == 'H':
            calculation_method = self.information_matrix_type_H
        else:
            raise ValueError(f'Unkown matrix_type {matrix_type}.')
        M = self._value_from_file_cache(filename, calculation_method)
        n = self.model_parameters_len
        assert M.shape == (n, n)
        return M

    def covariance_matrix_type_F_with_additional(self, include_variance_factor=True, **kwargs):
        information_matrix = self.information_matrix(matrix_type='F')
        information_matrix_increase = self.information_matrix_type_F_with_additional_increase_only(**kwargs)
        information_matrix += information_matrix_increase
        covariance_matrix = scipy.linalg.inv(information_matrix)
        if include_variance_factor:
            covariance_matrix *= self.variance_factor
        return covariance_matrix

    def covariance_matrix_type_F_with_additional_independent_increase_only(self, covariance_matrix, df_additional, standard_deviations_additional, include_variance_factor=True, dtype=None):
        if dtype is None:
            dtype = np.float128
        C = np.asarray(covariance_matrix, dtype=dtype)
        A = np.asarray(df_additional, dtype=dtype)
        D = np.asarray(standard_deviations_additional, dtype=dtype)
        U = A.T / D
        E = - C @ U @ scipy.linalg.inv(np.eye(U.shape[1]) + U.T @ C @ U) @ U.T @ C
        if include_variance_factor:
            E *= self.variance_factor
        return E

    def covariance_matrix_type_F_with_additional_independent(self, covariance_matrix, df_additional, standard_deviations_additional, include_variance_factor=True, dtype=None):
        if dtype is None:
            dtype = np.float128
        covariance_matrix = np.asarray(covariance_matrix, dtype=dtype)
        increase = self.covariance_matrix_type_F_with_additional_independent_increase_only(covariance_matrix, df_additional, standard_deviations_additional, include_variance_factor=include_variance_factor, dtype=dtype)
        covariance_matrix += increase
        return covariance_matrix

    def _covariance_matrix_calculate(self, matrix_type='F', include_variance_factor=True):
        util.logging.debug(f'Calculating model parameter covariance matrix of type {matrix_type} with include_variance_factor {include_variance_factor}.')

        if matrix_type == 'F':
            F = self.information_matrix(matrix_type='F')
            covariance_matrix = scipy.linalg.inv(F)
        elif matrix_type == 'H':
            H = self.information_matrix(matrix_type='H')
            covariance_matrix = scipy.linalg.inv(H)
        elif matrix_type == 'F_H':
            F = self.information_matrix(matrix_type='F')
            H = self.information_matrix(matrix_type='H')
            H_I = scipy.linalg.inv(H)
            covariance_matrix = H_I @ F @ H_I
        else:
            raise ValueError(f'Unkown matrix_type {matrix_type}.')

        if include_variance_factor:
            covariance_matrix *= self.variance_factor

        return covariance_matrix

    def covariance_matrix(self, matrix_type='F', include_variance_factor=True):
        filename = simulation.accuracy.constants.COVARIANCE_MATRIX_FILENAME.format(matrix_type=matrix_type, include_variance_factor=include_variance_factor)
        calculation_function = lambda: self._covariance_matrix_calculate(matrix_type=matrix_type, include_variance_factor=include_variance_factor)
        return self._value_from_file_cache(filename, calculation_function)

    def _correlation_matrix_calculate(self, matrix_type='F'):
        util.logging.debug(f'Calculating model parameter covariance matrix of type {matrix_type}.')
        covariance_matrix = self.covariance_matrix(matrix_type=matrix_type, include_variance_factor=False)
        inverse_standard_deviations = 1 / np.sqrt(covariance_matrix.diagonal())
        inverse_standard_deviations_diagonal_marix = np.diag(inverse_standard_deviations)
        correlation_matrix = inverse_standard_deviations_diagonal_marix @ covariance_matrix @ inverse_standard_deviations_diagonal_marix
        assert np.allclose(correlation_matrix.diagonal(), 1)
        for i in range(len(correlation_matrix)):
            correlation_matrix[i, i] = 1
        return correlation_matrix

    def correlation_matrix(self, matrix_type='F'):
        filename = simulation.accuracy.constants.CORRELATION_MATRIX_FILENAME.format(matrix_type=matrix_type)
        calculation_function = lambda: self._correlation_matrix_calculate(matrix_type=matrix_type)
        return self._value_from_file_cache(filename, calculation_function)

    def parameter_confidence_without_confidence_factor_using_covariance_matrix(self, covariance_matrix, relative=True):
        diagonal = covariance_matrix.diagonal()
        assert np.all(diagonal >= 0)
        confidences = np.sqrt(diagonal)
        if relative:
            confidences /= self.model_parameters
        return confidences

    def _parameter_confidence_calculate(self, alpha=0.99, relative=True,
                                        matrix_type='F', include_variance_factor=True, covariance_matrix=None):
        if covariance_matrix is None:
            covariance_matrix = self.covariance_matrix(matrix_type=matrix_type, include_variance_factor=include_variance_factor)
        confidences = self.parameter_confidence_without_confidence_factor_using_covariance_matrix(covariance_matrix, relative=relative)
        gamma = self.confidence_factor(alpha, include_variance_factor=include_variance_factor)
        return confidences * gamma

    def parameter_confidence(self, alpha=0.99, relative=True,
                             matrix_type='F', include_variance_factor=True, covariance_matrix=None):
        if covariance_matrix is not None:
            util.logging.debug(f'Calculating model parameter confidence for given covariance matrix with confidence level {alpha} and relative {relative}.')
            return self._parameter_confidence_calculate(
                matrix_type=matrix_type, include_variance_factor=include_variance_factor, covariance_matrix=covariance_matrix,
                alpha=alpha, relative=relative)
        else:
            util.logging.debug(f'Calculating model parameter confidence for covariance matrix of type {matrix_type} and include_variance_factor {include_variance_factor} with confidence level {alpha} and relative {relative}.')
            filename = simulation.accuracy.constants.PARAMETER_CONFIDENCE_FILENAME.format(matrix_type=matrix_type, include_variance_factor=include_variance_factor, alpha=alpha, relative=relative)
            calculation_function = lambda: self._parameter_confidence_calculate(matrix_type=matrix_type, include_variance_factor=include_variance_factor, alpha=alpha, relative=relative)
            return self._value_from_file_cache(filename, calculation_function)

    def average_parameter_confidence(self, alpha=0.99, relative=True,
                                     matrix_type='F', include_variance_factor=True, covariance_matrix=None, dtype=None):
        util.logging.debug(f'Calculating average model parameter confidence with confidence level {alpha} and relative {relative}.')
        if dtype is None:
            dtype = np.float64
        model_parameter_confidences = self.model_parameter_confidence(
            matrix_type=matrix_type, include_variance_factor=include_variance_factor, covariance_matrix=covariance_matrix,
            alpha=alpha, relative=relative)
        return np.nanmean(model_parameter_confidences, dtype=dtype)

    def average_parameter_standard_deviation(self, relative=True, matrix_type='F', include_variance_factor=True, covariance_matrix=None, dtype=None):
        util.logging.debug(f'Calculating average model parameter standrad deviation with relative {relative}.')
        if dtype is None:
            dtype = np.float64
        if covariance_matrix is None:
            covariance_matrix = self.covariance_matrix(matrix_type=matrix_type, include_variance_factor=include_variance_factor)
        standard_deviations = np.sqrt(np.diag(covariance_matrix))
        if relative:
            standard_deviations /= self.model_parameters
        return np.nanmean(standard_deviations, dtype=dtype)

    # *** model confidence *** #

    def _model_confidence_calculate_for_index(self, confidence_index, covariance_matrix, df_all, time_step_size):
        if not np.all(np.isnan(df_all[confidence_index])):
            time_index_start = confidence_index[1] * time_step_size
            # average
            confidence = 0.0
            for time_index_offset in range(time_step_size):
                df_i = df_all[confidence_index[0]][time_index_start + time_index_offset][confidence_index[2:]]
                assert df_i.ndim == 1
                confidence += np.sqrt(df_i @ covariance_matrix @ df_i)
            confidence /= time_step_size
        else:
            confidence = np.nan
        return confidence

    def model_confidence_without_confidence_factor_using_covariance_matrix(self, covariance_matrix, time_dim_confidence=12, df_all=None, time_dim_model=None, parallel=True, dtype=None):
        util.logging.debug(f'Calculating model confidence for given covariance matrix with confidence time dim {time_dim_confidence} and model time dim {time_dim_model} using parallel {parallel}.')
        # set dtype
        if dtype is None:
            dtype = np.float64
        # set time_dim_model and df_all
        if df_all is not None:
            time_dim_model = df_all.shape[1]
        else:
            if time_dim_model is None:
                time_dim_model = self.model.model_lsm.t_dim
            elif time_dim_model % time_dim_confidence != 0:
                raise ValueError(f'The desired time dimension {time_dim_confidence} of the confidence can not be satisfied because the time dimension of the model {time_dim_model} is not divisible by {time_dim_confidence}.')
            df_all = self.model_df_all_boxes(time_dim_model)
        # calculate time_step_size
        time_step_size = int(time_dim_model / time_dim_confidence)
        # calculate confidence shape
        confidence_shape = (df_all.shape[0], time_dim_confidence) + df_all.shape[2:-1]
        # calculate model confidence for each index
        if parallel < 1:
            model_confidence = np.empty(confidence_shape, dtype=dtype)
            for confidence_index in np.ndindex(*confidence_shape):
                model_confidence_at_index = self._model_confidence_calculate_for_index(
                    confidence_index, covariance_matrix, df_all, time_step_size)
                model_confidence[confidence_index] = model_confidence_at_index
        else:
            chunksize = np.sort(confidence_shape)[-1]
            covariance_matrix = util.parallel.with_multiprocessing.shared_array(covariance_matrix)
            df_all = util.parallel.with_multiprocessing.shared_array(df_all)
            model_confidence = util.parallel.with_multiprocessing.create_array_with_args(
                confidence_shape, self._model_confidence_calculate_for_index,
                covariance_matrix, df_all, time_step_size,
                number_of_processes=None, chunksize=chunksize, share_args=True)

        # return
        assert model_confidence.shape[1] == time_dim_confidence
        assert not np.all(np.isnan(model_confidence))
        return model_confidence

    def _model_confidence_calculate(self, alpha=0.99, time_dim_confidence=12, time_dim_model=None, parallel=True,
                                    matrix_type='F', include_variance_factor=True):
        covariance_matrix = self.covariance_matrix(matrix_type=matrix_type, include_variance_factor=include_variance_factor)
        confidences = self.model_confidence_without_confidence_factor_using_covariance_matrix(covariance_matrix, time_dim_confidence=time_dim_confidence, time_dim_model=time_dim_model, parallel=parallel)
        gamma = self.confidence_factor(alpha, include_variance_factor=include_variance_factor)
        return confidences * gamma

    def model_confidence(self, alpha=0.99, time_dim_confidence=12, time_dim_model=None, parallel=True,
                         matrix_type='F', include_variance_factor=True):
        if time_dim_model is None:
            time_dim_model = self.model.model_lsm.t_dim
        filename = simulation.accuracy.constants.MODEL_CONFIDENCE_FILENAME.format(
            matrix_type=matrix_type, include_variance_factor=include_variance_factor,
            alpha=alpha, time_dim_confidence=time_dim_confidence, time_dim_model=time_dim_model)
        calculation_function = lambda: self._model_confidence_calculate(
            matrix_type=matrix_type, include_variance_factor=include_variance_factor,
            alpha=alpha, time_dim_confidence=time_dim_confidence, time_dim_model=time_dim_model, parallel=parallel)
        return self._value_from_file_cache(filename, calculation_function, save_as_txt=False, save_as_np=True)

    # *** average model confidence *** #

    def average_model_confidence_using_model_confidence(self, model_confidence, per_tracer=False, relative=False, f_all=None, f_all_mean_per_tracer=None, dtype=None):
        util.logging.debug(f'Calculating average model confidence for given model confidence with per tracer {per_tracer} and relative {relative}.')
        if dtype is None:
            dtype = np.float128

        # averaging
        n = model_confidence.shape[0]
        average_model_confidence = np.empty(n, dtype=dtype)
        for i in range(n):
            average_model_confidence[i] = util.math.util.fnanmean(model_confidence[i])
        if relative:
            if f_all_mean_per_tracer is None:
                if f_all is None:
                    f_all = self.model_f_all_boxes(time_dim=1)
                f_all_mean_per_tracer = np.fromiter(map(util.math.util.fnanmean, f_all), dtype=dtype, count=len(f_all))
            average_model_confidence /= f_all_mean_per_tracer
        if not per_tracer:
            average_model_confidence = np.mean(average_model_confidence, dtype=dtype)

        # return
        assert not np.any(np.isnan(average_model_confidence))
        return average_model_confidence

    def _average_model_confidence_calculate(self, per_tracer=False, relative=False,
                                            alpha=0.99, time_dim_model=1, parallel=True,
                                            matrix_type='F', include_variance_factor=True):
        model_confidence = self.model_confidence(
            alpha=alpha, time_dim_confidence=1, time_dim_model=time_dim_model, parallel=parallel,
            matrix_type=matrix_type, include_variance_factor=include_variance_factor)
        return self.average_model_confidence_using_model_confidence(model_confidence, per_tracer=per_tracer, relative=relative)

    def average_model_confidence(self, per_tracer=False, relative=False,
                                 alpha=0.99, time_dim_model=1, parallel=True,
                                 matrix_type='F', include_variance_factor=True):
        if time_dim_model is None:
            time_dim_model = self.model.model_lsm.t_dim
        filename = simulation.accuracy.constants.AVERAGE_MODEL_CONFIDENCE_FILENAME.format(
            matrix_type=matrix_type, include_variance_factor=include_variance_factor,
            alpha=alpha, time_dim_model=time_dim_model,
            per_tracer=per_tracer, relative=relative)
        calculation_function = lambda: self._average_model_confidence_calculate(
            matrix_type=matrix_type, include_variance_factor=include_variance_factor,
            alpha=alpha, time_dim_model=time_dim_model,
            per_tracer=per_tracer, relative=relative)
        return self._value_from_file_cache(filename, calculation_function, save_as_txt=True, save_as_np=False)

    # *** average model confidence increase *** #

    def _confidence_increase_without_confidence_factor_calculate_for_index(self, confidence_index, confidence_type, df_all, covariance_matrix, number_of_measurements, relative,
                                                                           f_all_mean_per_tracer, include_variance_factor, dtype):
        df_i = df_all[confidence_index]
        if np.any(np.isfinite(df_i)):
            util.logging.debug(f'Calculating average model output confidence increase for index {confidence_index}.')
            # get standard deviation
            measurements_i = self.measurements.measurements_list[confidence_index[0]]
            coordinate = self.model.model_lsm.map_index_to_coordinate(*confidence_index[1:])
            index_measurements = measurements_i.sample_lsm.coordinate_to_map_index(*coordinate, discard_year=True)
            standard_deviations_additional = measurements_i.standard_deviations_for_sample_lsm()[tuple(index_measurements)]
            assert not np.any(np.isnan(standard_deviations_additional))
            # repeat several times if needed
            df_additional = np.tile(df_i, (number_of_measurements, 1))
            standard_deviations_additional = np.tile(standard_deviations_additional, number_of_measurements)
            # calculate confidence
            covariance_matrix = self.covariance_matrix_type_F_with_additional_independent(covariance_matrix, df_additional, standard_deviations_additional, include_variance_factor=include_variance_factor, dtype=dtype)
            if confidence_type == 'average_model_confidence':
                model_confidence_without_confidence_factor = self.model_confidence_without_confidence_factor_using_covariance_matrix(covariance_matrix, time_dim_confidence=1, df_all=df_all, parallel=False, dtype=dtype)
                confidence_increase_at_index = self.average_model_confidence_using_model_confidence(model_confidence_without_confidence_factor, per_tracer=False, relative=relative, f_all_mean_per_tracer=f_all_mean_per_tracer, dtype=dtype)
                assert confidence_increase_at_index is not None
            elif confidence_type == 'average_parameter_standard_deviation':
                confidence_increase_at_index = self.average_parameter_standard_deviation(covariance_matrix=covariance_matrix, relative=relative, dtype=dtype)
            else:
                assert False
        else:
            confidence_increase_at_index = np.nan
        return confidence_increase_at_index

    def _confidence_increase_calculate(self, confidence_type='average_model_confidence', number_of_measurements=1, alpha=0.99,
                                       relative=True, include_variance_factor=True, time_dim_model=None, parallel=True, dtype=None):
        util.logging.debug(f'Calculating confidence increase of type {confidence_type} with confidence level {alpha}, relative {relative}, time dim model {time_dim_model} and number_of_measurements {number_of_measurements} with include_variance_factor {include_variance_factor}.')
        if dtype is None:
            dtype = np.float128

        use_average_model_confidence = confidence_type == 'average_model_confidence'

        # get time_dim_model
        if time_dim_model is None:
            time_dim_model = self.model.model_lsm.t_dim

        # get needed values
        if use_average_model_confidence and relative:
            f_all = self.model_f_all_boxes(time_dim_model)
            f_all_mean_per_tracer = np.fromiter(map(util.math.util.fnanmean, f_all), dtype=dtype, count=len(f_all))
        else:
            f_all_mean_per_tracer = None
        df_all = self.model_df_all_boxes(time_dim_model)
        covariance_matrix = self.covariance_matrix(matrix_type='F', include_variance_factor=False)

        # make confidence_increase array
        confidence_increase_shape = (df_all.shape[0], time_dim_model) + df_all.shape[2:-1]

        # change time dim in model lsm
        model_lsm = self.model.model_lsm
        old_time_dim_model = model_lsm.t_dim
        model_lsm.t_dim = time_dim_model

        # calculate average_model_confidence increase for each index
        if not parallel:
            confidence_increase = np.empty(confidence_increase_shape, dtype=dtype)
            for confidence_index in np.ndindex(*confidence_increase_shape):
                confidence_increase_without_confidence_factor_at_index = self._confidence_increase_without_confidence_factor_calculate_for_index(
                    confidence_index, confidence_type, df_all, covariance_matrix, number_of_measurements, relative,
                    f_all_mean_per_tracer, False, dtype)
                confidence_increase[confidence_index] = confidence_increase_without_confidence_factor_at_index
        else:
            chunksize = np.sort(confidence_increase_shape)[-1]
            parallel = 0.5
            f_all_mean_per_tracer = util.parallel.with_multiprocessing.shared_array(f_all_mean_per_tracer)
            df_all = util.parallel.with_multiprocessing.shared_array(df_all)
            covariance_matrix = util.parallel.with_multiprocessing.shared_array(covariance_matrix)
            confidence_increase = util.parallel.with_multiprocessing.create_array_with_args(
                confidence_increase_shape, self._confidence_increase_without_confidence_factor_calculate_for_index,
                confidence_type, df_all, covariance_matrix, number_of_measurements, relative,
                f_all_mean_per_tracer, False, dtype,
                number_of_processes=None, chunksize=chunksize, share_args=True)

        # restore time dim in model lsm
        model_lsm.t_dim = old_time_dim_model

        # apply confidence factor and variance factor
        factor = 1
        if use_average_model_confidence:
            factor *= self.confidence_factor(alpha, include_variance_factor=include_variance_factor)
        if include_variance_factor:
            factor *= self.variance_factor**0.5
        confidence_increase *= factor

        # claculate increase of confidence
        if use_average_model_confidence:
            confidence = self.average_model_confidence(per_tracer=False, relative=relative,
                                                       alpha=alpha, time_dim_model=time_dim_model, parallel=parallel,
                                                       matrix_type='F', include_variance_factor=include_variance_factor)
        else:
            confidence = self.average_parameter_standard_deviation(relative=increases_calculation_relative, matrix_type='F', include_variance_factor=include_variance_factor)
        confidence_increase = confidence - confidence_increase
        return confidence_increase

    def confidence_increase(self, confidence_type='average_model_confidence', number_of_measurements=1, alpha=0.99,
                            time_dim_model=None, time_dim_confidence=12,
                            increases_calculation_relative=True, confidence_increases_relative_to_confidence=True,
                            include_variance_factor=True, parallel=True):
        if confidence_type not in ('average_model_confidence', 'average_parameter_standard_deviation'):
            raise ValueError(f'Confidence type {confidence_type} is unknown.')
        if time_dim_model is None:
            time_dim_model = self.model.model_lsm.t_dim
        # cache value
        filename = simulation.accuracy.constants.CONFIDENCE_INCREASE_FILENAME.format(
            confidence_type=confidence_type, number_of_measurements=number_of_measurements, alpha=alpha, relative=increases_calculation_relative,
            time_dim_model=time_dim_model, include_variance_factor=include_variance_factor)
        calculation_function = lambda: self._confidence_increase_calculate(
            confidence_type=confidence_type, number_of_measurements=number_of_measurements, alpha=alpha, relative=increases_calculation_relative,
            time_dim_model=time_dim_model, include_variance_factor=include_variance_factor,
            parallel=parallel)
        confidence_increase = self._value_from_file_cache(filename, calculation_function, save_as_txt=False, save_as_np=True)
        # change time_dim_confidence
        confidence_increase = measurements.plot.util.change_one_dim(confidence_increase, new_dim=time_dim_confidence, axis=1)
        # calculate relative
        if confidence_increases_relative_to_confidence:
            if confidence_type == 'average_model_confidence':
                confidence = self.average_model_confidence(per_tracer=False, relative=increases_calculation_relative,
                                                           alpha=alpha, time_dim_model=time_dim_model, parallel=parallel,
                                                           matrix_type='F', include_variance_factor=include_variance_factor)
            else:
                confidence = self.average_parameter_standard_deviation(relative=increases_calculation_relative, matrix_type='F', include_variance_factor=include_variance_factor)
            confidence_increase /= confidence
        # return
        assert not np.all(np.isnan(confidence_increase))
        return confidence_increase


class OLS(Base):

    @property
    def name(self):
        name = super().name
        standard_deviation_id = self.measurements.standard_deviation_id
        if len(standard_deviation_id) > 0:
            name = name + '(' + standard_deviation_id + ')'
        return name

    @property
    def _cost_function_class(self):
        return simulation.optimization.cost_function.OLS

    def _information_matrix_type_F_with_args(self, df, dtype=None):
        if dtype is None:
            dtype = np.float64
        assert df is not None
        df = np.asarray(df)
        assert df.ndim == 2
        # calculate matrix
        util.logging.debug(f'Calculating information matrix of type F for {self.name} with df {df.shape}.')
        average_standard_deviation = self.measurements.standard_deviations.mean(dtype=dtype)
        M = df.T @ df
        M *= (average_standard_deviation)**-2
        return M

    def information_matrix_type_F_with_additional_increase_only(self, **kwargs):
        df = kwargs['df']
        return self._information_matrix_type_F_with_args(df)

    def information_matrix_type_F(self):
        df = self.model_df(derivative_order=1)
        return self._information_matrix_type_F_with_args(df)


class WLS(Base):

    @property
    def name(self):
        name = super().name
        standard_deviation_id = self.measurements.standard_deviation_id
        if len(standard_deviation_id) > 0:
            name = name + '(' + standard_deviation_id + ')'
        return name

    @property
    def _cost_function_class(self):
        return simulation.optimization.cost_function.WLS

    def _information_matrix_type_F_with_args(self, df, standard_deviations):
        assert df is not None
        assert standard_deviations is not None
        df = np.asarray(df)
        standard_deviations = np.asarray(standard_deviations)
        assert df.ndim == 2
        assert standard_deviations.ndim == 1
        assert len(df) == len(standard_deviations)
        # calculate matrix
        util.logging.debug(f'Calculating information matrix of type F for {self.name} with df {df.shape}.')
        weighted_df = df * standard_deviations[:, np.newaxis]**-1
        M = weighted_df.T @ weighted_df
        return M

    def information_matrix_type_F_with_additional_increase_only(self, **kwargs):
        df = kwargs['df']
        standard_deviations = kwargs['standard_deviations']
        return self._information_matrix_type_F_with_args(df, standard_deviations)

    def information_matrix_type_F(self):
        df = self.model_df(derivative_order=1)
        standard_deviations = self.measurements.standard_deviations
        return self._information_matrix_type_F_with_args(df, standard_deviations)


class GLS(Base):

    @property
    def name(self):
        name = super().name
        correlation_id = self.measurements.correlation_id
        if len(correlation_id) > 0:
            name = name + '(' + correlation_id + ')'
        return name

    @property
    def _cost_function_class(self):
        return simulation.optimization.cost_function.GLS

    def _information_matrix_type_F_with_args(self, df, standard_deviations, correlation_matrix=None, correlation_matrix_decomposition=None, dtype=None):
        if dtype is None:
            dtype = np.float64
        assert df is not None
        assert standard_deviations is not None
        assert correlation_matrix is not None or correlation_matrix_decomposition is not None
        df = np.asarray(df, dtype=dtype)
        standard_deviations = np.asarray(standard_deviations, dtype=dtype)
        assert df.ndim == 2
        assert standard_deviations.ndim == 1
        assert len(df) == len(standard_deviations)
        if correlation_matrix_decomposition is None:
            assert correlation_matrix.ndim == 2
            assert correlation_matrix.shape[0] == correlation_matrix.shape[1]
            assert correlation_matrix.shape[0] == standard_deviations.shape[0]
            correlation_matrix_decomposition = matrix.decompose(correlation_matrix, return_type=matrix.LDL_DECOMPOSITION_TYPE)
        # calculate matrix
        util.logging.debug(f'Calculating information matrix of type F for {self.name} with df {df.shape}.')
        weighted_df = df * standard_deviations[:, np.newaxis]**-1
        M = correlation_matrix_decomposition.inverse_matrix_both_sides_multiplication(weighted_df)
        return M

    def information_matrix_type_F_with_additional_increase_only(self, **kwargs):
        df = kwargs['df']
        standard_deviations = kwargs['standard_deviations']
        correlation_matrix = kwargs['correlation_matrix']
        return self._information_matrix_type_F_with_args(df, standard_deviations, correlation_matrix=correlation_matrix)

    def information_matrix_type_F(self):
        df = self.model_df(derivative_order=1)
        standard_deviations = self.measurements.standard_deviations
        correlation_matrix_decomposition = self.measurements.correlations_own_decomposition
        return self._information_matrix_type_F_with_args(df, standard_deviations, correlation_matrix_decomposition=correlation_matrix_decomposition)
