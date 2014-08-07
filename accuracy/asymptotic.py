import os.path
import scipy.stats
import logging
import numpy as np

import ndop.util.value_cache
import ndop.util.data_base

from util.math.matrix import SingularMatrixError
import util.parallel

from .constants import CACHE_DIRNAME, INFORMATION_MATRIX_FILENAME, COVARIANCE_MATRIX_FILENAME, PARAMETER_CONFIDENCE_FILENAME, MODEL_CONFIDENCE_FILENAME, AVERAGE_MODEL_CONFIDENCE_FILENAME, AVERAGE_MODEL_CONFIDENCE_INCREASE_FILENAME, PROJECTED_DF_FILENAME



class Base():
    
    def __init__(self, data_kind, spinup_options, time_step=1, df_accuracy_order=2, job_setup=None):
        cf_kind = self.__class__.__name__
        kind = data_kind + '_' + cf_kind
        
        if job_setup is None:
            job_setup = {}
        try:
            job_setup['name']
        except KeyError:
            job_setup['name'] = 'A_' + kind
        
        cache_dirname = os.path.join(CACHE_DIRNAME, kind)
        self.cache = ndop.util.value_cache.Cache(spinup_options, time_step, df_accuracy_order=df_accuracy_order, cache_dirname=cache_dirname)
        self.data_base = ndop.util.data_base.init_data_base(data_kind, spinup_options, time_step=time_step, df_accuracy_order=df_accuracy_order, job_setup=job_setup)
    
    
    
    def information_matrix_calculate(self, parameters, DF_additional=None, inverse_variances_additional=None):
        raise NotImplementedError("Please implement this method")
    
    def information_matrix(self, parameters, DF_additional=None, inverse_variances_additional=None):
        if DF_additional is None:
            return self.cache.get_value(parameters, INFORMATION_MATRIX_FILENAME, self.information_matrix_calculate, derivative_used=True, save_also_txt=True)
        else:
            return self.information_matrix_calculate(parameters, DF_additional=DF_additional, inverse_variances_additional=inverse_variances_additional)
    
    
    
    def covariance_matrix_calculate(self, parameters, DF_additional=None, inverse_variances_additional=None):
        if DF_additional is None:
            logging.debug('Calculating covariance matrix.')
        else:
            logging.debug('Calculating covariance matrix with {} additional measurements.'.format(len(DF_additional)))
        
        information_matrix = np.matrix(self.information_matrix(parameters, DF_additional=DF_additional, inverse_variances_additional=inverse_variances_additional))
        try:
            information_matrix = information_matrix.I
        except np.linalg.linalg.LinAlgError as exc:
            raise SingularMatrixError(information_matrix) from  exc
#         information_matrix = np.array(information_matrix)
        return information_matrix
    
    def covariance_matrix(self, parameters, DF_additional=None, inverse_variances_additional=None):
        if DF_additional is None:
            return self.cache.get_value(parameters, COVARIANCE_MATRIX_FILENAME, self.covariance_matrix_calculate, derivative_used=True, save_also_txt=True)
        else:
            return self.covariance_matrix_calculate(parameters, DF_additional=DF_additional, inverse_variances_additional=inverse_variances_additional)
    
    
    
    def parameter_confidence_calculate(self, parameters, alpha=0.99):
        logging.debug('Calculating parameter confidence with confidence level {}.'.format(alpha))
        
        C = self.covariance_matrix(parameters)
        d = np.diag(C)
        n = C.shape[0]
        
        gamma = scipy.stats.chi2.ppf(alpha, n)
        confidences = d**(1/2) * gamma**(1/2)
        
        return confidences
    
    def parameter_confidence(self, parameters):
        return self.cache.get_value(parameters, PARAMETER_CONFIDENCE_FILENAME, self.parameter_confidence_calculate, derivative_used=True, save_also_txt=True)
    
    
    
    def model_confidence_calculate(self, parameters, DF_additional=None, inverse_variances_additional=None, alpha=0.99):
        if DF_additional is None:
            logging.debug('Calculating model confidence.')
        else:
            logging.debug('Calculating model confidence with {} additional measurements.'.format(len(DF_additional)))
        
        C = np.asmatrix(self.covariance_matrix(parameters, DF_additional=DF_additional, inverse_variances_additional=inverse_variances_additional))
        n = C.shape[0]
        gamma = scipy.stats.chi2.ppf(alpha, n)
        
        df_all = self.data_base.df_all(parameters)
        confidence = np.empty(df_all.shape[:-1])
        
        for i in np.ndindex(*df_all.shape[:-1]):
            df_i = df_all[i]
            if not any(np.isnan(df_i)):
                df_i = np.matrix(df_i, copy=False).T
                confidence[i] = df_i.T * C * df_i
            else:
                confidence[i] = np.nan
        
        mask = np.logical_not(np.isnan(confidence))
        confidence[mask] = confidence[mask]**(1/2) * gamma**(1/2)
        
        return confidence
    
    def model_confidence(self, parameters, DF_additional=None, inverse_variances_additional=None):
        if DF_additional is None:
            return self.cache.get_value(parameters, MODEL_CONFIDENCE_FILENAME, self.model_confidence_calculate, derivative_used=True, save_also_txt=False)
        else:
            return self.model_confidence_calculate(parameters, DF_additional=DF_additional, inverse_variances_additional=inverse_variances_additional)
    
    
    
    def average_model_confidence_calculate(self, parameters, DF_additional=None, inverse_variances_additional=None):
        if DF_additional is None:
            logging.debug('Calculating average model confidence.')
        else:
            logging.debug('Calculating average model confidence with {} additional measurements.'.format(len(DF_additional)))
        
        model_confidence = self.model_confidence(parameters, DF_additional=DF_additional, inverse_variances_additional=inverse_variances_additional)
        
        mask = np.logical_not(np.isnan(model_confidence))
        average_model_confidence = model_confidence[mask].mean()
        
        return average_model_confidence
    
    def average_model_confidence(self, parameters, DF_additional=None, inverse_variances_additional=None):
        if DF_additional is None:
            return self.cache.get_value(parameters, AVERAGE_MODEL_CONFIDENCE_FILENAME, self.average_model_confidence_calculate, derivative_used=True, save_also_txt=True)
        else:
            return self.average_model_confidence_calculate(parameters, DF_additional=DF_additional, inverse_variances_additional=inverse_variances_additional)
    
    
    
    def average_model_confidence_increase_calculate_for_index(self, parameters, index):
        logging.debug('Calculating average model confidence increase for index {}.'.format(index))
        
        ## get necessary values
        df_all = self.data_base.df_all(parameters)
        inverse_variances_all = self.data_base.inverse_variances_all
        
        ## compute increse for index
        df_index = df_all[index]
        if not any(np.isnan(df_index)):
            DF_additional = [df_index, df_index]
            DF_additional[int(not index[0])] = df_index * 0
            inverse_variances_additional = [inverse_variances_all[0][index[1:]], inverse_variances_all[1][index[1:]]]
            
            average_model_confidence_increase_index = self.average_model_confidence(parameters, DF_additional=DF_additional, inverse_variances_additional=inverse_variances_additional)
        else:
            average_model_confidence_increase_index = np.nan
        
        return average_model_confidence_increase_index
    
    
    def average_model_confidence_increase_calculate(self, parameters, ):
        logging.debug('Calculating average model confidence increase.')
        
        df_all = self.data_base.df_all(parameters)
        
        average_model_confidence_increase = util.parallel.create_array(df_all.shape[:-1], self.average_model_confidence_increase_calculate_for_index, function_args=(parameters,), function_args_first=True, chunksize=16)
        
        average_model_confidence = self.average_model_confidence(parameters)
        average_model_confidence_increase -= average_model_confidence
        return average_model_confidence_increase
    
    
#     def average_model_confidence_increase_calculate(self, parameters):
#         logging.debug('Calculating average model confidence increase.')
#         
#         df_all = self.data_base.df_all(parameters)
#         inverse_variances_all = self.data_base.inverse_variances_all
#         average_model_confidence = self.average_model_confidence(parameters)
#         
#         average_model_confidence_increase = np.empty(df_all.shape[:-1])
#         
#         for i in np.ndindex(*df_all.shape[:-1]):
#             df_i = df_all[i]
#             if not any(np.isnan(df_i)):
#                 DF_additional = [df_i, df_i]
#                 DF_additional[int(not i[0])] = df_i * 0
#                 inverse_variances_additional = [inverse_variances_all[0][i[1:]], inverse_variances_all[1][i[1:]]]
#                 
#                 average_model_confidence_increase[i] = self.average_model_confidence(parameters, DF_additional=DF_additional, inverse_variances_additional=inverse_variances_additional)
#             else:
#                 average_model_confidence_increase[i] = np.nan
#         
#         average_model_confidence_increase -= average_model_confidence
#         return average_model_confidence_increase
    
    
    def average_model_confidence_increase(self, parameters):
        return self.cache.get_value(parameters, AVERAGE_MODEL_CONFIDENCE_INCREASE_FILENAME, self.average_model_confidence_increase_calculate, derivative_used=True, save_also_txt=False)



class OLS(Base):
    
    def information_matrix_calculate(self, parameters, DF_additional=None, inverse_variances_additional=None):
        data_base = self.data_base
        
        ## fix measurements
        if DF_additional is None:
            DF = data_base.DF(parameters)
            n = DF.shape[-1]
            
            M = np.zeros([n, n], dtype=np.float128)
            for i in np.ndindex(*DF.shape[:-1]):
                M += np.outer(DF[i], DF[i])
            
            M *= data_base.average_inverse_variance
        
        ## additional measurements
        else:
            M = self.information_matrix(parameters)
            for tracer_i in range(len(DF_additional)):
                DF_additional_tracer = DF_additional[tracer_i]
                for i in np.ndindex(*DF_additional_tracer.shape[:-1]):
                    M += np.outer(DF_additional_tracer[i], DF_additional_tracer[i]) * data_base.average_inverse_variance
        
        return M


class WLS(Base):
    
    def information_matrix_calculate(self, parameters, DF_additional=None, inverse_variances_additional=None):
        data_base = self.data_base
        
        ## fix measurements
        if DF_additional is None:
            DF = data_base.DF(parameters)
            n = DF.shape[-1]
            inverse_variances = data_base.inverse_variances
            
            M = np.zeros([n, n], dtype=np.float128)
            for i in np.ndindex(*DF.shape[:-1]):
                M += np.outer(DF[i], DF[i]) * inverse_variances[i]
        
        ## additional measurements
        else:
            M = self.information_matrix(parameters)
            for tracer_i in range(len(DF_additional)):
                DF_additional_tracer = DF_additional[tracer_i]
                inverse_variances_additional_tracer = inverse_variances_additional[tracer_i]
                for i in np.ndindex(*DF_additional_tracer.shape[:-1]):
                    M += np.outer(DF_additional_tracer[i], DF_additional_tracer[i]) * inverse_variances_additional_tracer[i]
        
        return M


class GLS(Base):
    
    def projected_DF_calculate(self, parameters, DF_additional=None, inverse_variances_additional=None):
        data_base = self.data_base
        
        ## fix measurements
        if DF_additional is None:
            DF = data_base.DF(parameters)
            DF = data_base.inverse_deviations[:, np.newaxis] * DF
            projected_DF = data_base.project(DF)
            projected_DF = np.array(projected_DF)
        
        ## additional measurements
        else:
            projected_DF = self.projected_DF(parameters)
            for tracer_i in range(len(DF_additional)):
                DF_additional[tracer_i] = inverse_deviations_additional[tracer_i][:, np.newaxis] * DF_additional[tracer_i]
            projected_DF_additional = np.array(data_base.project(DF_additional))
            
            for i in np.ndindex(*projected_DF.shape[:2]):
                projected_DF[i] += projected_DF_additional[i]
        
        return projected_DF
    
    def projected_DF(self, parameters, DF_additional=None, inverse_variances_additional=None):
        if DF_additional is None:
            return self.cache.get_value(parameters, PROJECTED_DF_FILENAME, self.projected_DF_calculate, derivative_used=True)
        else:
            return self.projected_DF_calculate(parameters, DF_additional=DF_additional, inverse_variances_additional=inverse_variances_additional)
    
    
    def information_matrix_calculate(self, parameters, DF_additional=None, inverse_variances_additional=None):
        projected_DF = self.projected_DF(parameters, DF_additional=DF_additional, inverse_variances_additional=inverse_variances_additional)
        M = self.data_base.projected_product_inverse_correlation_matrix_both_sides(projected_DF[0], projected_DF[1], correlation_parameter)
        
        return M
    
    
#     def information_matrix_calculate(self, parameters):
#         data_base = self.data_base
#         DF = data_base.DF(parameters)
#         correlation_parameters = data_base.correlation_parameters(parameters)
#         
#         M = data_base.product_inverse_covariance_matrix_both_sides(DF, correlation_parameters)
#         
#         return M


 


class Family(ndop.util.data_base.Family):
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

