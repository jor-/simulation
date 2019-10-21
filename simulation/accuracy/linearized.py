import math

import numpy as np
import scipy.linalg
import scipy.stats

import matrix

import util.logging
import util.parallel.with_multiprocessing

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

    def model_confidence_without_confidence_factor_using_covariance_matrix(self, covariance_matrix, time_dim_confidence=12, time_dim_model=None, parallel=True, df_all=None, dtype=None):
        util.logging.debug(f'Calculating model confidence for given covariance matrix with confidence time dim {time_dim_confidence} and model time dim {time_dim_model} using parallel {parallel}.')
        if dtype is None:
            dtype = np.float64

        # calculate needed values
        if time_dim_model is None:
            time_dim_model = self.model.model_lsm.t_dim
        if time_dim_model % time_dim_confidence == 0:
            time_step_size = int(time_dim_model / time_dim_confidence)
        else:
            raise ValueError(f'The desired time dimension {time_dim_confidence} of the confidence can not be satisfied because the time dimension of the model {time_dim_model} is not divisible by {time_dim_confidence}.')

        if df_all is None:
            df_all = self.model_df_all_boxes(time_dim_model)
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

    def average_model_confidence_using_model_confidence(self, model_confidence, per_tracer=False, relative=False, f_all=None, dtype=None):
        util.logging.debug(f'Calculating average model confidence for given model confidence with per tracer {per_tracer} and relative {relative}.')
        if dtype is None:
            dtype = np.float128

        # averaging
        def fnanmean(a):
            a = a[~ np.isnan(a)]
            sum = math.fsum(a)
            mean = sum / len(a)
            return mean

        n = model_confidence.shape[0]
        average_model_confidence = np.empty(n, dtype=dtype)
        for i in range(n):
            average_model_confidence[i] = fnanmean(model_confidence[i])
        if relative:
            if f_all is None:
                f_all = self.model_f_all_boxes(time_dim=1)
            for i in range(n):
                average_model_confidence[i] /= fnanmean(f_all[i])
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

    def _average_model_confidence_increase_without_confidence_factor_calculate_for_index(self, confidence_index, f_all, df_all, covariance_matrix, number_of_measurements, relative, include_variance_factor, parallel, dtype):
        time_dim_model = df_all.shape[1]
        df_additional = df_all[confidence_index]
        if not np.all(np.isnan(df_additional)):
            util.logging.debug(f'Calculating average model output confidence increase for index {confidence_index}.')
            # get standard deviation
            coordinate = self.model.model_lsm.map_index_to_coordinate(*confidence_index[1:])
            measurements_i = self.measurements.measurements_list[confidence_index[0]]
            index_measurements = measurements_i.sample_lsm.coordinate_to_map_index(*coordinate, discard_year=True)
            standard_deviations_additional = measurements_i.standard_deviations_for_sample_lsm()[tuple(index_measurements)]
            assert not np.any(np.isnan(standard_deviations_additional))
            # repeat several times if needed
            df_additional = np.tile(df_additional, (number_of_measurements, 1))
            standard_deviations_additional = np.tile(standard_deviations_additional, number_of_measurements)
            # calculate confidence
            covariance_matrix = self.covariance_matrix_type_F_with_additional_independent(covariance_matrix, df_additional, standard_deviations_additional, include_variance_factor=include_variance_factor, dtype=dtype)
            model_confidence_without_confidence_factor = self.model_confidence_without_confidence_factor_using_covariance_matrix(covariance_matrix, time_dim_confidence=1, time_dim_model=time_dim_model, parallel=False, df_all=df_all, dtype=dtype)
            average_model_confidence_increase_without_confidence_factor_at_index = self.average_model_confidence_using_model_confidence(model_confidence_without_confidence_factor, per_tracer=False, relative=relative, f_all=f_all, dtype=dtype)
            assert average_model_confidence_increase_without_confidence_factor_at_index is not None
        else:
            average_model_confidence_increase_without_confidence_factor_at_index = np.nan
        return average_model_confidence_increase_without_confidence_factor_at_index

    def _average_model_confidence_increase_calculate(self, number_of_measurements=1, alpha=0.99, time_dim_confidence_increase=12, time_dim_model=None, relative=True, include_variance_factor=True, parallel=True, dtype=None):
        util.logging.debug(f'Calculating average model output confidence increase with confidence level {alpha}, relative {relative}, model time dim {time_dim_model}, condifence time dim {time_dim_confidence_increase} and number_of_measurements {number_of_measurements} with include_variance_factor {include_variance_factor}.')
        if dtype is None:
            dtype = np.float128

        # get needed values
        if relative:
            f_all = self.model_f_all_boxes(time_dim_model)
        else:
            f_all = None
        df_all = self.model_df_all_boxes(time_dim_model)
        covariance_matrix = self.covariance_matrix(matrix_type='F', include_variance_factor=False)

        # make average_model_confidence_increase array
        average_model_confidence_increase_shape = (df_all.shape[0], time_dim_confidence_increase) + df_all.shape[2:-1]

        # change time dim in model lsm
        model_lsm = self.model.model_lsm
        old_time_dim_model = model_lsm.t_dim
        model_lsm.t_dim = time_dim_model

        # calculate average_model_confidence increase for each index
        if not parallel:
            average_model_confidence_increase = np.empty(average_model_confidence_increase_shape, dtype=dtype)
            for confidence_index in np.ndindex(*average_model_confidence_increase_shape):
                average_model_confidence_increase_without_confidence_factor_at_index = self._average_model_confidence_increase_without_confidence_factor_calculate_for_index(
                    confidence_index, f_all, df_all, covariance_matrix, number_of_measurements, relative, False, parallel, dtype)
                average_model_confidence_increase[confidence_index] = average_model_confidence_increase_without_confidence_factor_at_index
        else:
            chunksize = np.sort(average_model_confidence_increase_shape)[-1]
            parallel = 0.5
            if f_all is not None:
                f_all = util.parallel.with_multiprocessing.shared_array(f_all)
            df_all = util.parallel.with_multiprocessing.shared_array(df_all)
            covariance_matrix = util.parallel.with_multiprocessing.shared_array(covariance_matrix)
            average_model_confidence_increase = util.parallel.with_multiprocessing.create_array_with_args(
                average_model_confidence_increase_shape, self._average_model_confidence_increase_without_confidence_factor_calculate_for_index,
                f_all, df_all, covariance_matrix, number_of_measurements, relative, False, parallel, dtype,
                number_of_processes=None, chunksize=chunksize, share_args=True)

        # restore time dim in model lsm
        model_lsm.t_dim = old_time_dim_model

        # apply confidence factor and variance factor
        factor = self.confidence_factor(alpha, include_variance_factor=include_variance_factor)
        if include_variance_factor:
            factor *= self.variance_factor**0.5
        average_model_confidence_increase *= factor

        # claculate increase of confidence
        average_model_confidence = self.average_model_confidence(per_tracer=False, relative=relative,
                                                                 alpha=alpha, time_dim_model=time_dim_model, parallel=parallel,
                                                                 matrix_type='F', include_variance_factor=include_variance_factor)
        average_model_confidence_increase = average_model_confidence - average_model_confidence_increase
        return average_model_confidence_increase

    def average_model_confidence_increase(self, number_of_measurements=1, alpha=0.99, time_dim_confidence_increase=12, time_dim_model=None, relative=True, include_variance_factor=True, parallel=True):
        if time_dim_model is None:
            time_dim_model = self.model.model_lsm.t_dim

        filename = simulation.accuracy.constants.AVERAGE_MODEL_CONFIDENCE_INCREASE_FILENAME.format(
            number_of_measurements=number_of_measurements, alpha=alpha, relative=relative,
            time_dim_confidence_increase=time_dim_confidence_increase, time_dim_model=time_dim_model,
            include_variance_factor=include_variance_factor)
        calculation_function = lambda: self._average_model_confidence_increase_calculate(
            number_of_measurements=number_of_measurements, alpha=alpha, relative=relative,
            time_dim_confidence_increase=time_dim_confidence_increase, time_dim_model=time_dim_model,
            include_variance_factor=include_variance_factor, parallel=parallel)
        average_model_confidence_increase = self._value_from_file_cache(filename, calculation_function, save_as_txt=False, save_as_np=True)

        assert not np.all(np.isnan(average_model_confidence_increase))
        return average_model_confidence_increase


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
