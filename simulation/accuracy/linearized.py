import math

import numpy as np
import scipy.linalg
import scipy.stats

import matrix

import util.cache.memory
import util.logging
import util.parallel.with_multiprocessing

import simulation.accuracy.constants
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
        self.dtype = np.float128

    def model_df(self):
        df = super().model_df(derivative_kind=None).astype(self.dtype)
        assert df.shape == (self.measurements.number_of_measurements, self.model_parameters_len)
        return df

    # *** uncertainty model parameters *** #

    @staticmethod
    def confidence_factor(alpha):
        assert 0 < alpha < 1
        return scipy.stats.norm.ppf((1 + alpha) / 2)

    def _model_parameter_information_matrix_calculate(self, **kwargs):
        raise NotImplementedError("Please implement this method")

    @util.cache.memory.method_decorator()
    def model_parameter_information_matrix(self, **kwargs):
        if len(kwargs):
            M = self._model_parameter_information_matrix_calculate(**kwargs)
        else:
            M = self._value_from_file_cache(simulation.accuracy.constants.INFORMATION_MATRIX_FILENAME,
                                            self._model_parameter_information_matrix_calculate)
        n = self.model_parameters_len
        assert M.shape == (n, n) and M.dtype == self.dtype
        return M

    def _model_parameter_covariance_matrix_calculate(self,
                                                     information_matrix=None):
        util.logging.debug('Calculating model parameter covariance matrix.')
        if information_matrix is None:
            information_matrix = self.model_parameter_information_matrix()
        else:
            information_matrix = np.asarray(information_matrix, dtype=self.dtype)
        covariance_matrix = scipy.linalg.inv(information_matrix)
        return covariance_matrix

    @util.cache.memory.method_decorator()
    def model_parameter_covariance_matrix(self,
                                          information_matrix=None):
        if information_matrix is not None:
            return self._model_parameter_covariance_matrix_calculate(information_matrix=information_matrix)
        else:
            return self._value_from_file_cache(simulation.accuracy.constants.COVARIANCE_MATRIX_FILENAME,
                                               self._model_parameter_covariance_matrix_calculate)

    def model_parameter_correlation_matrix(self,
                                           information_matrix=None):
        util.logging.debug('Calculating model parameter correlation matrix.')
        covariance_matrix = self.model_parameter_covariance_matrix(information_matrix=information_matrix)
        inverse_derivatives = np.sqrt(covariance_matrix.diagonal())
        inverse_derivative_diagonal_marix = np.diag(inverse_derivatives)
        correlation_matrix = inverse_derivative_diagonal_marix @ covariance_matrix @ inverse_derivative_diagonal_marix
        return correlation_matrix

    def _model_parameter_confidence_calculate(self, alpha=0.99, relative=True,
                                              information_matrix=None, model_parameter_covariance_matrix=None):
        util.logging.debug(f'Calculating model parameter confidence with confidence level {alpha} and relative {relative}.')
        if model_parameter_covariance_matrix is None:
            model_parameter_covariance_matrix = self.model_parameter_covariance_matrix(information_matrix=information_matrix)
        diagonal = model_parameter_covariance_matrix.diagonal()
        gamma = self.confidence_factor(alpha)
        confidences = np.sqrt(diagonal) * gamma
        if relative:
            confidences /= self.model_parameters
        return confidences

    def model_parameter_confidence(self, alpha=0.99, relative=True,
                                   information_matrix=None, model_parameter_covariance_matrix=None):
        if information_matrix is not None or model_parameter_covariance_matrix is not None:
            return self._model_parameter_confidence_calculate(alpha=alpha, relative=relative, information_matrix=information_matrix, model_parameter_covariance_matrix=model_parameter_covariance_matrix)
        else:
            return self._value_from_file_cache(simulation.accuracy.constants.PARAMETER_CONFIDENCE_FILENAME.format(alpha=alpha, relative=relative),
                                               lambda: self._model_parameter_confidence_calculate(alpha=alpha, relative=relative))

    def average_model_parameter_confidence(self, alpha=0.99, relative=True,
                                           information_matrix=None, model_parameter_covariance_matrix=None):
        util.logging.debug(f'Calculating average model parameter confidence with confidence level {alpha} and relative {relative}.')
        return self.model_parameter_confidence(alpha=alpha, relative=relative, information_matrix=information_matrix, model_parameter_covariance_matrix=model_parameter_covariance_matrix).mean(dtype=self.dtype)

    # *** uncertainty in model output *** #

    def _model_confidence_calculate_for_index(self, confidence_index, model_parameter_covariance_matrix, df_all, time_step_size, gamma):
        if not np.all(np.isnan(df_all[confidence_index])):
            time_index_start = confidence_index[1] * time_step_size
            # average
            confidence = 0.0
            for time_index_offset in range(time_step_size):
                df_i = df_all[confidence_index[0]][time_index_start + time_index_offset][confidence_index[2:]]
                assert df_i.ndim == 1
                confidence += np.sqrt(df_i @ model_parameter_covariance_matrix @ df_i)
            confidence /= time_step_size
            # mutiply with confidence factor
            confidence *= gamma
        else:
            confidence = np.nan
        return confidence

    def _model_confidence_calculate(self, alpha=0.99, time_dim_confidence=12, time_dim_model=None, parallel=True,
                                    information_matrix=None, model_parameter_covariance_matrix=None):
        util.logging.debug(f'Calculating model confidence with confidence level {alpha}, desired time dim {time_dim_confidence} of the confidence and time dim {time_dim_model}.')

        # calculate needed values
        if time_dim_model % time_dim_confidence == 0:
            time_step_size = int(time_dim_model / time_dim_confidence)
        else:
            raise ValueError(f'The desired time dimension {time_dim_confidence} of the confidence can not be satisfied because the time dimension of the model {time_dim_model} is not divisible by {time_dim_confidence}.')

        if model_parameter_covariance_matrix is None:
            model_parameter_covariance_matrix = self.model_parameter_covariance_matrix(information_matrix=information_matrix)
        df_all = self.model_df_all_boxes(time_dim_model, as_shared_array=parallel > 0)
        gamma = self.confidence_factor(alpha)
        confidence_shape = (df_all.shape[0], time_dim_confidence) + df_all.shape[2:-1]

        # calculate model confidence for each index
        if parallel < 1:
            model_confidence = np.empty(confidence_shape, dtype=self.dtype)
            for confidence_index in np.ndindex(*confidence_shape):
                model_confidence_at_index = self._model_confidence_calculate_for_index(
                    confidence_index,
                    model_parameter_covariance_matrix, df_all, time_step_size, gamma)
                model_confidence[confidence_index] = model_confidence_at_index
        else:
            chunksize = np.sort(confidence_shape)[-1]
            model_confidence = util.parallel.with_multiprocessing.create_array_with_args(
                confidence_shape, self._model_confidence_calculate_for_index,
                model_parameter_covariance_matrix, df_all, time_step_size, gamma,
                number_of_processes=None, chunksize=chunksize, share_args=True)

        # return
        return model_confidence

    def model_confidence(self, alpha=0.99, time_dim_confidence=12, time_dim_model=None, parallel=True,
                         information_matrix=None, model_parameter_covariance_matrix=None):
        if time_dim_model is None:
            time_dim_model = self.model.model_lsm.t_dim
        if information_matrix is not None or model_parameter_covariance_matrix is not None:
            model_confidence = self._model_confidence_calculate(alpha=alpha, parallel=parallel,
                                                                time_dim_confidence=time_dim_confidence, time_dim_model=time_dim_model,
                                                                information_matrix=information_matrix, model_parameter_covariance_matrix=model_parameter_covariance_matrix)
        else:
            model_confidence = self._value_from_file_cache(simulation.accuracy.constants.MODEL_CONFIDENCE_FILENAME.format(
                alpha=alpha, time_dim_confidence=time_dim_confidence, time_dim_model=time_dim_model),
                lambda: self._model_confidence_calculate(
                alpha=alpha, time_dim_confidence=time_dim_confidence, time_dim_model=time_dim_model, parallel=parallel),
                save_as_txt=False, save_as_np=True)

        assert model_confidence.shape[1] == time_dim_confidence
        assert not np.all(np.isnan(model_confidence))
        return model_confidence

    def _average_model_confidence_calculate(self, alpha=0.99, time_dim_model=None, per_tracer=False, relative=True, parallel=True,
                                            information_matrix=None, model_parameter_covariance_matrix=None):
        util.logging.debug(f'Calculating average model output confidence with confidence level {alpha}, per_tracer {per_tracer}, relative {relative} and model time dim {time_dim_model}.')
        # model confidence
        if information_matrix is None and model_parameter_covariance_matrix is None:
            time_dim_confidence = 12
        else:
            time_dim_confidence = 1
        model_confidence = self.model_confidence(alpha=alpha, parallel=parallel,
                                                 time_dim_confidence=time_dim_confidence, time_dim_model=time_dim_model,
                                                 information_matrix=information_matrix, model_parameter_covariance_matrix=model_parameter_covariance_matrix)

        # averaging
        def fnanmean(a):
            a = a[~ np.isnan(a)]
            sum = math.fsum(a)
            mean = sum / len(a)
            return mean

        n = model_confidence.shape[0]
        average_model_confidence = np.empty(n, dtype=self.dtype)
        for i in range(n):
            average_model_confidence[i] = fnanmean(model_confidence[i])
        if relative:
            model_output = self.model_f_all_boxes(time_dim_model, as_shared_array=parallel)
            for i in range(n):
                average_model_confidence[i] /= fnanmean(model_output[i])
        if not per_tracer:
            average_model_confidence = np.mean(average_model_confidence, dtype=self.dtype)

        util.logging.debug(f'Average model confidence {average_model_confidence} calculated for confidence level {alpha} and model time dim {time_dim_model} using relative values {relative}.')
        return average_model_confidence

    def average_model_confidence(self, alpha=0.99, time_dim_model=None, per_tracer=False, relative=True, parallel=True,
                                 information_matrix=None, model_parameter_covariance_matrix=None):
        if time_dim_model is None:
            time_dim_model = self.model.model_lsm.t_dim

        if information_matrix is not None or model_parameter_covariance_matrix is not None:
            average_model_confidence = self._average_model_confidence_calculate(
                alpha=alpha, time_dim_model=time_dim_model,
                per_tracer=per_tracer, relative=relative, parallel=parallel,
                information_matrix=information_matrix, model_parameter_covariance_matrix=model_parameter_covariance_matrix)
        else:
            average_model_confidence = self._value_from_file_cache(simulation.accuracy.constants.AVERAGE_MODEL_CONFIDENCE_FILENAME.format(
                alpha=alpha, time_dim_model=time_dim_model,
                per_tracer=per_tracer, relative=relative),
                lambda: self._average_model_confidence_calculate(
                alpha=alpha, time_dim_model=time_dim_model,
                per_tracer=per_tracer, relative=relative, parallel=parallel))
        assert not np.any(np.isnan(average_model_confidence))
        return average_model_confidence

    def _average_model_confidence_increase_calculate_for_index(self, confidence_index, df_all, number_of_measurements, alpha, time_dim_model, relative, parallel):
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
            if number_of_measurements > 1:
                df_additional = np.tile(df_additional, number_of_measurements)
                standard_deviations_additional = np.tile(standard_deviations_additional, number_of_measurements)
            # calculate confidence
            model_parameter_covariance_matrix_additional_independent = self.model_parameter_covariance_matrix_additional_independent(df_additional, standard_deviations_additional)
            average_model_confidence_increase_at_index = self.average_model_confidence(alpha=alpha, time_dim_model=time_dim_model, per_tracer=False, relative=relative, parallel=False, model_parameter_covariance_matrix=model_parameter_covariance_matrix_additional_independent)
            assert average_model_confidence_increase_at_index is not None
        else:
            average_model_confidence_increase_at_index = np.nan
        return average_model_confidence_increase_at_index

    def _average_model_confidence_increase_calculate(self, number_of_measurements=1, alpha=0.99, time_dim_confidence_increase=12, time_dim_model=None, relative=True, parallel=True):
        util.logging.debug(f'Calculating average model output confidence increase with confidence level {alpha}, relative {relative}, model time dim {time_dim_model}, condifence time dim {time_dim_confidence_increase} and number_of_measurements {number_of_measurements}.')

        # get all df
        df_all = self.model_df_all_boxes(time_dim_model, as_shared_array=parallel)

        # make average_model_confidence_increase array
        average_model_confidence_increase_shape = (df_all.shape[0], time_dim_confidence_increase) + df_all.shape[2:-1]

        # change time dim in model lsm
        model_lsm = self.model.model_lsm
        old_time_dim_model = model_lsm.t_dim
        model_lsm.t_dim = time_dim_model

        # calculate average_model_confidence increase for each index
        if not parallel:
            average_model_confidence_increase = np.empty(average_model_confidence_increase_shape, dtype=self.dtype)
            for confidence_index in np.ndindex(*average_model_confidence_increase_shape):
                average_model_confidence_increase_at_index = self._average_model_confidence_increase_calculate_for_index(
                    confidence_index, df_all, number_of_measurements, alpha, time_dim_model, relative, parallel)
                average_model_confidence_increase[index] = average_model_confidence_increase_at_index
        else:
            chunksize = np.sort(average_model_confidence_increase_shape)[-1]
            parallel = 0.5
            average_model_confidence_increase = util.parallel.with_multiprocessing.map_parallel_with_args(
                self._average_model_confidence_increase_calculate_for_index, np.ndindex(*average_model_confidence_increase_shape),
                df_all, number_of_measurements, alpha, time_dim_model, relative, parallel,
                number_of_processes=None, chunksize=chunksize, share_args=True)

        # restore time dim in model lsm
        model_lsm.t_dim = old_time_dim_model

        # claculate increase of confidence
        average_model_confidence = self.average_model_confidence(alpha=alpha, time_dim_model=time_dim_model, relative=relative, parallel=parallel)
        average_model_confidence_increase = average_model_confidence - average_model_confidence_increase
        return average_model_confidence_increase

    def average_model_confidence_increase(self, number_of_measurements=1, alpha=0.99, time_dim_confidence_increase=12, time_dim_model=None, relative=True, parallel=True):
        if time_dim_model is None:
            time_dim_model = self.model.model_lsm.t_dim

        average_model_confidence_increase = self._value_from_file_cache(simulation.accuracy.constants.AVERAGE_MODEL_CONFIDENCE_INCREASE_FILENAME.format(
            number_of_measurements=number_of_measurements, alpha=alpha, relative=relative,
            time_dim_confidence_increase=time_dim_confidence_increase, time_dim_model=time_dim_model),
            lambda: self._average_model_confidence_increase_calculate(
            number_of_measurements=number_of_measurements, alpha=alpha, relative=relative,
            time_dim_confidence_increase=time_dim_confidence_increase, time_dim_model=time_dim_model, parallel=parallel),
            save_as_txt=False, save_as_np=True)

        assert not np.all(np.isnan(average_model_confidence_increase))
        return average_model_confidence_increase

    # *** additional methods *** #

    def model_parameter_information_matrix_additional_independent_increase(self, df_additional, standard_deviations_additional):
        correlation_matrix = scipy.sparse.eye(len(standard_deviations_additional))
        model_parameter_information_matrix = self.model_parameter_information_matrix(df=df_additional, standard_deviations=standard_deviations_additional, correlation_matrix=correlation_matrix)
        return model_parameter_information_matrix

    def model_parameter_information_matrix_additional_independent(self, df_additional, standard_deviations_additional):
        matrix = self.model_parameter_information_matrix()
        increase = self.model_parameter_information_matrix_additional_independent_increase(df_additional, standard_deviations_additional)
        return matrix + increase

    def model_parameter_covariance_matrix_additional_independent_increase(self, df_additional, standard_deviations_additional):
        A = np.asarray(df_additional, dtype=self.dtype)
        C = self.model_parameter_covariance_matrix()
        D = np.asarray(standard_deviations_additional, dtype=self.dtype)
        if D.size > 1:
            D = np.diag(D)
            E = - C @ A.T @ np.linalg.inv(D + A @ C @ A.T) @ A @ C
        else:
            v = C @ A
            E = - np.outer(v, v) / (D + A @ C @ A)
        return E

    def model_parameter_covariance_matrix_additional_independent(self, df_additional, standard_deviations_additional):
        matrix = self.model_parameter_covariance_matrix()
        increase = self.model_parameter_covariance_matrix_additional_independent_increase(df_additional, standard_deviations_additional)
        return matrix + increase


class OLS(Base):

    def _model_parameter_information_matrix_calculate(self, df=None):
        # prepare df
        if df is None:
            df = self.model_df().astype(self.dtype)
        else:
            df = np.asarray(df, dtype=self.dtype)
        assert df.ndim == 2
        util.logging.debug(f'Calculating information matrix of type {self.name} with df {df.shape}.')
        # calculate matrix
        average_standard_deviation = self.measurements.standard_deviations.mean(dtype=self.dtype)
        M = df.T @ df
        M *= (average_standard_deviation)**-2
        return M

    def model_parameter_information_matrix(self, **kwargs):
        M = super().model_parameter_information_matrix()
        if len(kwargs) > 0:
            M += self._model_parameter_information_matrix_calculate(df=kwargs['df'])


class WLS(Base):

    def _model_parameter_information_matrix_calculate(self, df=None, standard_deviations=None):
        # prepare df and standard deviations
        if df is None:
            df = self.model_df().astype(self.dtype)
            standard_deviations = self.measurements.standard_deviations
        else:
            assert standard_deviations is not None
            df = np.asarray(df, dtype=self.dtype)
            standard_deviations = np.asarray(standard_deviations, dtype=self.dtype)
        assert df.ndim == 2
        assert standard_deviations.ndim == 1
        assert len(df) == len(standard_deviations)
        # calculate matrix
        util.logging.debug(f'Calculating information matrix of type {self.name} with df {df.shape}.')
        weighted_df = df * standard_deviations[:, np.newaxis]**-1
        M = weighted_df.T @ weighted_df
        return M

    def model_parameter_information_matrix(self, **kwargs):
        M = super().model_parameter_information_matrix()
        if len(kwargs) > 0:
            M += self._model_parameter_information_matrix_calculate(df=kwargs['df'], standard_deviations=kwargs['standard_deviations'])


class GLS(Base):

    def _model_parameter_information_matrix_calculate(self, df=None, standard_deviations=None, correlation_matrix=None, correlation_matrix_decomposition=None):
        # prepare df and standard deviations and correlation matrix decomposition
        if df is None:
            df = self.model_df().astype(self.dtype)
            standard_deviations = self.measurements.standard_deviations
            correlation_matrix_decomposition = self.measurements.correlations_own_decomposition
        else:
            assert standard_deviations is not None
            assert correlation_matrix is not None or correlation_matrix_decomposition is not None
            df = np.asarray(df, dtype=self.dtype)
            standard_deviations = np.asarray(standard_deviations, dtype=self.dtype)
            if correlation_matrix_decomposition is None:
                correlation_matrix = np.asarray(correlation_matrix, dtype=self.dtype)
                assert correlation_matrix.ndim == 2
                assert correlation_matrix.shape[0] == correlation_matrix.shape[1]
                assert correlation_matrix.shape[0] == standard_deviations.shape[0]
                correlation_matrix_decomposition = matrix.decompose(correlation_matrix, return_type=matrix.LDL_DECOMPOSITION_TYPE)
        assert df.ndim == 2
        assert standard_deviations.ndim == 1
        assert len(df) == len(standard_deviations)
        # calculate matrix
        util.logging.debug(f'Calculating information matrix of type {self.name} with df {df.shape}.')
        weighted_df = df / standard_deviations[:, np.newaxis]
        M = correlation_matrix_decomposition.inverse_matrix_both_sides_multiplication(weighted_df)
        return M
