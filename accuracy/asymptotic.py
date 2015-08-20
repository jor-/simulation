import os.path
import scipy.stats
import numpy as np

import ndop.util.value_cache
import ndop.util.data_base

from util.math.matrix import SingularMatrixError
import util.parallel.universal
import util.parallel.with_multiprocessing

import util.logging
logger = util.logging.logger

from .constants import CACHE_DIRNAME, INFORMATION_MATRIX_FILENAME, COVARIANCE_MATRIX_FILENAME, PARAMETER_CONFIDENCE_FILENAME, MODEL_CONFIDENCE_FILENAME, AVERAGE_MODEL_CONFIDENCE_FILENAME, AVERAGE_MODEL_CONFIDENCE_INCREASE_FILENAME


class Base():
    
    def __init__(self, data_kind, spinup_options=ndop.util.data_base.DEFAULT_SPINUP_OPTIONS, time_step=1, df_accuracy_order=2, job_setup=None):
        cf_kind = self.__class__.__name__
        kind = data_kind + '_' + cf_kind
        
        if job_setup is None:
            job_setup = {}
        try:
            job_setup['name']
        except KeyError:
            job_setup['name'] = 'A_' + kind
        
        cache_dirname = os.path.join(CACHE_DIRNAME, kind)
        self.cache = ndop.util.value_cache.Cache(spinup_options, time_step, df_accuracy_order=df_accuracy_order, cache_dirname=cache_dirname, use_memory_cache=True)
        self.data_base = ndop.util.data_base.init_data_base(data_kind, spinup_options, time_step=time_step, df_accuracy_order=df_accuracy_order, job_setup=job_setup)
        
        self.dtype = np.float128
    
    
    
    def information_matrix_calculate(self, parameters, additionals=None):
        raise NotImplementedError("Please implement this method")
    
    def information_matrix(self, parameters, additionals=None):
        if additionals is None:
            return self.cache.get_value(parameters, INFORMATION_MATRIX_FILENAME, self.information_matrix_calculate, derivative_used=True, save_also_txt=True)
        else:
            return self.information_matrix_calculate(parameters, additionals)
    
    
    
    
    def covariance_matrix_calculate_with_information_matrix(self, information_matrix):
        logger.debug('Calculating covariance matrix for information matrix.')
        
        information_matrix = np.asmatrix(information_matrix, dtype=self.dtype)
        try:
            covariance_matrix = information_matrix.I
        except np.linalg.linalg.LinAlgError as exc:
            raise SingularMatrixError(information_matrix) from exc
        return covariance_matrix
    
    
    def covariance_matrix_calculate_with_parameters(self, parameters):
        logger.debug('Calculating covariance matrix for parameters.')
        information_matrix = self.information_matrix(parameters)
        return self.covariance_matrix_calculate_with_information_matrix(information_matrix)
    
    
    def covariance_matrix(self, parameters_or_information_matrix):
        if parameters_or_information_matrix.ndim == 2:
            information_matrix = parameters_or_information_matrix
            return self.covariance_matrix_calculate_with_information_matrix(information_matrix)
        elif parameters_or_information_matrix.ndim == 1:
            parameters = parameters_or_information_matrix
            return self.cache.get_value(parameters, COVARIANCE_MATRIX_FILENAME, self.covariance_matrix_calculate_with_parameters, derivative_used=True, save_also_txt=True)
        else:
            raise ValueError('Wrong shape: parameters_or_information_matrix must have 1 or 2 dimensions but it has {} dimensions.'.format(parameters_or_information_matrix.ndim))
    
    
    
    def parameter_confidence_calculate(self, parameters_or_information_matrix, alpha=0.99):
        covariance_matrix = self.covariance_matrix(parameters_or_information_matrix)
        
        logger.debug('Calculating parameter confidence with confidence level {}.'.format(alpha))
        
        C = np.asmatrix(covariance_matrix)
        d = np.diag(C)
        n = C.shape[0]
        
        gamma = scipy.stats.chi2.ppf(alpha, n)
        confidences = d**(1/2) * gamma**(1/2)
        
        logger.debug('Parameter confidence calculated.')
        
        return confidences
    
    
    def parameter_confidence(self, parameters_or_information_matrix):
        if parameters_or_information_matrix.ndim == 2:
            information_matrix = parameters_or_information_matrix
            return self.parameter_confidence_calculate(information_matrix)
        elif parameters_or_information_matrix.ndim == 1:
            parameters = parameters_or_information_matrix
            return self.cache.get_value(parameters, PARAMETER_CONFIDENCE_FILENAME, self.parameter_confidence_calculate, derivative_used=True, save_also_txt=True)
        else:
            raise ValueError('Wrong shape: parameters_or_information_matrix must have 1 or 2 dimensions but it has {} dimensions.'.format(parameters_or_information_matrix.ndim))
    
    
    def model_confidence_calculate_for_index(self, confidence_index, C, df_boxes, time_step_size, gamma, mask_is_sea, value_mask=None):
        if mask_is_sea[confidence_index[2:]] and (value_mask is None or value_mask[confidence_index]):
            ## average
            confidence = 0
            for df_time_index in range(time_step_size):
                df_i = df_boxes[confidence_index[0]][confidence_index[1]*time_step_size + df_time_index][confidence_index[2:]]
                df_i = np.matrix(df_i, copy=True).T
                confidence += df_i.T * C * df_i
            confidence /= time_step_size
            
            confidence = confidence.item()**(1/2) * gamma**(1/2)
        else:
            confidence = np.nan
        
#         logger.debug('Model confidence {} calculated for index {}.'.format(confidence, confidence_index))
        
        return confidence
    
    
    
    def model_confidence_calculate(self, parameters, information_matrix=None, alpha=0.99, time_dim_confidence=12, time_dim_df=2880, value_mask=None, use_mem_map=False, parallel_mode=util.parallel.universal.max_parallel_mode()):
        logger.debug('Calculating model confidence with confidence level {}, desired time dim {} of the confidence and time dim {} of df.'.format(alpha, time_dim_confidence, time_dim_df))
        
        ## calculate time step size
        if time_dim_df % time_dim_confidence == 0:
            time_step_size = int(time_dim_df / time_dim_confidence)
        else:
            raise ValueError('The desired time dimension {0} of the confidence can not be satisfied because the time dimension of df {1} is not divisible by {0}.'.format(time_dim_confidence, time_dim_df))
        
        ## calculate covariance matrix
        if information_matrix is not None:
            covariance_matrix = self.covariance_matrix(information_matrix)
        else:
            covariance_matrix = self.covariance_matrix(parameters)
        C = np.asmatrix(covariance_matrix)
        
        ## calculate confidence level
        n = C.shape[0]
        gamma = scipy.stats.chi2.ppf(alpha, n)
        
        ## calculate df_boxes, value_mask and mask_is_sea
        as_shared_array = parallel_mode == util.parallel.universal.MODES['multiprocessing']
        df_boxes = self.data_base.df_boxes(parameters, time_dim=time_dim_df, use_memmap=use_mem_map, as_shared_array=as_shared_array)
        mask_is_sea = ~ np.isnan(df_boxes[0,0,:,:,:,0])
        if as_shared_array:
            value_mask = util.parallel.with_multiprocessing.shared_array(value_mask)
            mask_is_sea = util.parallel.with_multiprocessing.shared_array(mask_is_sea)
        
        ## init confidence
        confidence_shape = (df_boxes.shape[0], time_dim_confidence) + df_boxes.shape[2:-1]
        confidence = np.empty(confidence_shape)
        
        if value_mask is not None:
            assert confidence_shape == value_mask.shape
        
        ## calculate confidence
        confidence = util.parallel.universal.create_array(confidence_shape, self.model_confidence_calculate_for_index, C, df_boxes, time_step_size, gamma, mask_is_sea, value_mask, parallel_mode=parallel_mode, chunksize=128*64)
        
        return confidence
    
    
    def model_confidence(self, parameters, information_matrix=None, time_dim_confidence=12, time_dim_df=2880, value_mask=None, use_mem_map=False, parallel_mode=util.parallel.universal.max_parallel_mode()):
        if information_matrix is None:
            return self.cache.get_value(parameters, MODEL_CONFIDENCE_FILENAME.format(time_dim_confidence=time_dim_confidence, time_dim_df=time_dim_df), lambda p: self.model_confidence_calculate(p, time_dim_confidence=time_dim_confidence, time_dim_df=time_dim_df, use_mem_map=use_mem_map, parallel_mode=parallel_mode), derivative_used=True, save_also_txt=False)
        else:
            return self.model_confidence_calculate(parameters, information_matrix, time_dim_confidence=time_dim_confidence, time_dim_df=time_dim_df, value_mask=value_mask, use_mem_map=use_mem_map, parallel_mode=parallel_mode)
    
    
    
    def average_model_confidence_calculate(self, parameters, information_matrix=None, time_dim_df=2880, value_mask=None, use_mem_map=False, parallel_mode=util.parallel.universal.max_parallel_mode()):
        
        model_confidence = self.model_confidence(parameters, information_matrix, time_dim_confidence=12, time_dim_df=time_dim_df, value_mask=value_mask, use_mem_map=use_mem_map, parallel_mode=parallel_mode)
        
        if value_mask is not None:
            model_confidence = model_confidence[value_mask]
#         average_model_confidence = model_confidence[np.logical_not(np.isnan(model_confidence))].mean(dtype=self.dtype)
        average_model_confidence = np.nanmean(model_confidence, dtype=self.dtype)
        
        alpha = 0.99
        logger.debug('Average model confidence {} calculated for confidence level {} and time dim {} of df.'.format(average_model_confidence, alpha, time_dim_df))
        
        return average_model_confidence
    
    
    def average_model_confidence(self, parameters, information_matrix=None, time_dim_df=2880, value_mask=None, use_mem_map=False, parallel_mode=util.parallel.universal.max_parallel_mode()):
        if information_matrix is None and value_mask is None:
            return self.cache.get_value(parameters, AVERAGE_MODEL_CONFIDENCE_FILENAME.format(time_dim_df=time_dim_df), lambda p: self.average_model_confidence_calculate(p, time_dim_df=time_dim_df, value_mask=value_mask, use_mem_map=use_mem_map, parallel_mode=parallel_mode), derivative_used=True, save_also_txt=True)
        else:
            return self.average_model_confidence_calculate(parameters, information_matrix, time_dim_df=time_dim_df, value_mask=value_mask, use_mem_map=use_mem_map, parallel_mode=parallel_mode)
    
    
    
    def average_model_confidence_increase_calculate_for_index(self, index, parameters, number_of_measurements=1, time_dim_df=2880, value_mask=None, use_mem_map=False, parallel_mode=util.parallel.universal.max_parallel_mode()):
        ## get necessary values
        df_boxes_increase = self.data_base.df_boxes(parameters, time_dim=12)
        inverse_deviation_boxes_increase = self.data_base.inverse_deviations_boxes(time_dim=12)
        
        ## compute increse for index
        if not any(np.isnan(df_boxes_increase[index])):
            additional = {'DF': np.tile(df_boxes_increase[index][np.newaxis].T, number_of_measurements).T, 'inverse_deviations': np.tile(inverse_deviation_boxes_increase[index], number_of_measurements), 'split_index': int(not index[0]) * number_of_measurements}
            information_matrix = self.information_matrix(parameters, additional)
            average_model_confidence_increase_index = self.average_model_confidence(parameters, information_matrix, time_dim_df=time_dim_df, value_mask=value_mask, use_mem_map=use_mem_map, parallel_mode=parallel_mode)
        else:
            average_model_confidence_increase_index = np.nan
        
        logger.debug('Average model confidence {} calulated for index {}.'.format(average_model_confidence_increase_index, index))
        return average_model_confidence_increase_index
    
    
    
    def average_model_confidence_increase_calculate(self, parameters, number_of_measurements=1, time_dim_df=2880, value_mask=None, use_mem_map=False, parallel_mode=util.parallel.universal.max_parallel_mode()):
        logger.debug('Calculating average model confidence increase for parameters {} with {} additional measurements and time dim {} of df.'.format(parameters, number_of_measurements, time_dim_df))
        
        ## set parallel mode and share arrays
        if parallel_mode == util.parallel.universal.MODES['scoop']:
            parallel_mode_average_model_confidence_increase = util.parallel.universal.MODES['scoop']
            parallel_mode_average_model_confidence = util.parallel.universal.MODES['multiprocessing']
            parallel_mode_average_model_confidence_last = util.parallel.universal.MODES['multiprocessing']
        elif parallel_mode == util.parallel.universal.MODES['multiprocessing']:
#             parallel_mode_average_model_confidence_increase = util.parallel.universal.MODES['serial']
#             parallel_mode_average_model_confidence = util.parallel.universal.MODES['multiprocessing']
            parallel_mode_average_model_confidence_increase = util.parallel.universal.MODES['multiprocessing']
            parallel_mode_average_model_confidence = util.parallel.universal.MODES['serial']
            parallel_mode_average_model_confidence_last = util.parallel.universal.MODES['multiprocessing']
            
            ## create shared arrays
            value_mask = util.parallel.with_multiprocessing.shared_array(value_mask)
            self.data_base.df_boxes(parameters, time_dim=time_dim_df, as_shared_array=True)
            self.data_base.df_boxes(parameters, time_dim=12, as_shared_array=True)
            self.data_base.inverse_deviations_boxes(time_dim=12, as_shared_array=True)
        else:
            parallel_mode_average_model_confidence_increase = util.parallel.universal.MODES['serial']
            parallel_mode_average_model_confidence = util.parallel.universal.MODES['serial']
            parallel_mode_average_model_confidence_last = util.parallel.universal.MODES['serial']
        
        ## calculate
        df_boxes_increase = self.data_base.df_boxes(parameters, time_dim=12)
        df_boxes_confidence = self.data_base.df_boxes(parameters, time_dim=time_dim_df)
        assert df_boxes_increase.ndim == 6
        logger.debug('Calculating average model confidence increase for {} values.'.format(np.sum(~ np.isnan(df_boxes_increase))))
        
#         average_model_confidence_increase = util.parallel.universal.create_array(df_boxes.shape[:-1], self.average_model_confidence_increase_calculate_for_index_zipped, args=(parameters, time_dim_df, value_mask, False), parallel_mode=parallel_mode)
        average_model_confidence_increase = util.parallel.universal.create_array(df_boxes_increase.shape[:-1], self.average_model_confidence_increase_calculate_for_index, parameters, number_of_measurements, time_dim_df, value_mask, use_mem_map, parallel_mode_average_model_confidence, parallel_mode=parallel_mode_average_model_confidence_increase)
        
        average_model_confidence = self.average_model_confidence(parameters, time_dim_df=time_dim_df, value_mask=value_mask, use_mem_map=use_mem_map, parallel_mode=parallel_mode_average_model_confidence_last)
        average_model_confidence_increase = average_model_confidence - average_model_confidence_increase
        return average_model_confidence_increase
    
    
    def average_model_confidence_increase(self, parameters, number_of_measurements=1, time_dim_df=2880, value_mask=None, use_mem_map=False, parallel_mode=util.parallel.universal.max_parallel_mode()):
        if value_mask is None:
            return self.cache.get_value(parameters, AVERAGE_MODEL_CONFIDENCE_INCREASE_FILENAME.format(number_of_measurements=number_of_measurements, time_dim_df=time_dim_df), lambda p: self.average_model_confidence_increase_calculate(p, number_of_measurements=number_of_measurements, time_dim_df=time_dim_df, use_mem_map=use_mem_map, parallel_mode=parallel_mode), derivative_used=True, save_also_txt=False)
        else:
            return self.average_model_confidence_increase_calculate(parameters, number_of_measurements=number_of_measurements, time_dim_df=time_dim_df, value_mask=value_mask, use_mem_map=use_mem_map, parallel_mode=parallel_mode)






class OLS(Base):
    
    def information_matrix_calculate_with_DF(self, DF, inverse_average_variance):
        logger.debug('Calculating information matrix of type {} with {} DF values.'.format(self.__class__.__name__, len(DF)))
        
        assert DF.ndim == 2
        
        n, m = DF.shape
        M = np.zeros([m, m], dtype=self.dtype)
        
        for i in range(n):
            M += np.outer(DF[i], DF[i])
        M *= inverse_average_variance   
        
        assert M.ndim == 2 and M.shape[0] == M.shape[1] == DF.shape[1] and M.dtype == self.dtype
        return M
    
    
    def information_matrix_calculate_with_parameters(self, parameters):
        logger.debug('Calculating information matrix of type {} for parameters {}.'.format(self.__class__.__name__, parameters))
        
        DF = self.data_base.DF(parameters)
        M = self.information_matrix_calculate_with_DF(DF, self.data_base.inverse_average_variance)   
             
        assert M.ndim == 2 and M.shape[0] == M.shape[1] == len(parameters) and M.dtype == self.dtype
        return M
    
    
    def information_matrix_calculate(self, parameters, additionals=None):
        if additionals is None:
            return self.information_matrix_calculate_with_parameters(parameters)
        else:
            return self.information_matrix(parameters) + self.information_matrix_calculate_with_DF(additionals['DF'], self.data_base.inverse_average_variance)



class WLS(Base):
    
    def information_matrix_calculate_with_DF(self, DF, inverse_deviations):
        logger.debug('Calculating information matrix of type {} with {} DF values.'.format(self.__class__.__name__, len(DF)))
        
        assert DF.ndim == 2
        
        n, m = DF.shape
        M = np.zeros([m, m], dtype=self.dtype)
        
        for i in range(n):
            M += np.outer(DF[i], DF[i]) * inverse_deviations[i]**2
        
        assert M.ndim == 2 and M.shape[0] == M.shape[1] == DF.shape[1] and M.dtype == self.dtype
        return M
    
    
    def information_matrix_calculate_with_parameters(self, parameters):
        logger.debug('Calculating information matrix of type {} for parameters {}.'.format(self.__class__.__name__, parameters))
        
        DF = self.data_base.DF(parameters)
        M = self.information_matrix_calculate_with_DF(DF, self.data_base.inverse_deviations)
        
        assert M.ndim == 2 and M.shape[0] == M.shape[1] == len(parameters) and M.dtype == self.dtype
        return M
    
    
    def information_matrix_calculate(self, parameters, additionals=None):
        if additionals is None:
            return self.information_matrix_calculate_with_parameters(parameters)
        else:
            return self.information_matrix(parameters) + self.information_matrix_calculate_with_DF(additionals['DF'], additionals['inverse_deviations'])
    


class GLS(Base):
    
    def __init__(self, *args, correlation_min_values=10, correlation_max_year_diff=float('inf'), **kargs):        
        ## save additional kargs
        self.correlation_min_values = correlation_min_values
        if correlation_max_year_diff is None or correlation_max_year_diff < 0:
            correlation_max_year_diff = float('inf')
        self.correlation_max_year_diff = correlation_max_year_diff
        
        ## super init
        super().__init__(*args, **kargs)
    
    
    def information_matrix_calculate_with_DF(self, DF, correlation_matrix):
        logger.debug('Calculating information matrix of type {} with {} DF values.'.format(self.__class__.__name__, len(DF)))
        
        assert DF.ndim == 2 and correlation_matrix.ndim == 2
        assert correlation_matrix.shape[1] == DF.shape[0]
        
        DF = np.asmatrix(DF, dtype=self.dtype)
        correlation_matrix = np.asmatrix(correlation_matrix, dtype=self.dtype)
        
        M = DF.T * correlation_matrix.I * DF
        
        assert M.ndim == 2 and M.shape[0] == M.shape[1] == DF.shape[1] and M.dtype == self.dtype
        return M
    
    
    def information_matrix_calculate_with_parameters(self, parameters):
        P, L = self.data_base.correlation_matrix_cholesky_decomposition(self.correlation_min_values, self.correlation_max_year_diff)
        DF = self.data_base.DF(parameters)
        inverse_deviations = self.data_base.inverse_variances
        weighted_DF = DF * inverse_deviations[:, np.newaxis]
        weighted_DF = P * np.matrix(weighted_DF)
        
        X = util.math.sparse.solve.forward_substitution(L, weighted_DF, dtype=self.dtype)
        X = np.asmatrix(X)
        M = X.T * X
        
        assert M.ndim == 2 and M.shape[0] == M.shape[1] == len(parameters) and M.dtype == self.dtype
        return
    
    
    
    def information_matrix_calculate(self, parameters, additionals=None):
        if additionals is None:
            return self.information_matrix_calculate_with_parameters(parameters)
        else:
            return self.information_matrix(parameters) + self.information_matrix_calculate_with_DF(additionals['DF'], additionals['inverse_deviations'])
    
    


class GLS_P3(Base):
    
    def DF_projected_calculate_with_DF(self, DF, inverse_deviations, split_index, projected_value_index=0):
        logger.debug('Calculating projected DF {} with {} DF values.'.format(projected_value_index, len(DF)))
        
        DF = DF * inverse_deviations
        return self.data_base.project(DF, split_index, projected_value_index=projected_value_index)
    
    
    def DF_projected_calculate_with_parameters(self, parameters, projected_value_index=0):
        logger.debug('Calculating projected DF {} for parameters {}.'.format(projected_value_index, parameters))
        
        n = self.data_base.m_dop
        DF = self.data_base.DF(parameters)
        inverse_deviations = self.data_base.inverse_deviations
        
        return self.DF_projected_calculate_with_DF(DF, inverse_deviations, n, projected_value_index=projected_value_index)
    
    
    def DF_projected_calculate(self, parameters, additional=None, projected_value_index=0):
        if additional is None:
            return self.DF_projected_calculate_with_parameters(parameters, projected_value_index=projected_value_index)
        else:
            return self.DF_projected_calculate(parameters, projected_value_index=projected_value_index) + self.DF_projected_calculate_with_DF(additional['DF'], additional['inverse_deviations'], additional['split_index'], projected_value_index=projected_value_index)
    
    
    def DF_projected(self, parameters, additional=None, projected_value_index=None):
        if projected_value_index is not None:
            if additional is None:
                calculation_function = lambda p: self.DF_projected_calculate(p, projected_value_index=projected_value_index)
                return self.cache.get_value(parameters, PROJECTED_DF_FILENAME[projected_value_index], calculation_function, derivative_used=True)
            else:
                return self.DF_projected_calculate(parameters, additional, projected_value_index=projected_value_index)
        else:
            return [self.DF_projected(parameters, additional, projected_value_index=0), self.DF_projected(parameters, additional, projected_value_index=1)]
    
    
    def information_matrix_calculate(self, parameters, additional=None):
        DF_projected = self.DF_projected(parameters, additional)
        correlation_parameters = self.data_base.correlation_parameters(parameters)
        M = self.data_base.projected_product_inverse_correlation_matrix_both_sides(DF_projected, correlation_parameters)
        
        return M


 


class Family(ndop.optimization.data_base.Family):
    def __init__(self, main_member_class, data_kind, spinup_options, time_step=1, df_accuracy_order=2, job_setup=None):
        
        if data_kind.upper() == 'WOA':
            member_classes = (OLS, WLS)
        elif data_kind.upper() == 'WOD':
            member_classes = (OLS, WLS, GLS)
        else:
            raise ValueError('Data_kind {} unknown. Must be "WOA" or "WOD".'.format(data_kind))
        
        super().__init__(main_member_class, member_classes, data_kind, spinup_options, time_step=time_step, df_accuracy_order=df_accuracy_order, job_setup=job_setup)
    
    
    def information_matrix(self, parameters):
        fun = lambda o: o.information_matrix(parameters) 
        value = self.get_function_value(fun)
        return value

    def covariance_matrix(self, parameters):
        fun = lambda o: o.covariance_matrix(parameters) 
        value = self.get_function_value(fun)
        return value
    
    def parameter_confidence(self, parameters):
        fun = lambda o: o.parameter_confidence(parameters) 
        value = self.get_function_value(fun)
        return value
    
    def model_confidence(self, parameters):
        fun = lambda o: o.model_confidence(parameters) 
        value = self.get_function_value(fun)
        return value
    
    def average_model_confidence(self, parameters):
        fun = lambda o: o.parameter_confidence(parameters) 
        value = self.get_function_value(fun)
        return value
    
    def average_model_confidence_increase(self, parameters):
        fun = lambda o: o.average_model_confidence_increase(parameters) 
        value = self.get_function_value(fun)
        return value



