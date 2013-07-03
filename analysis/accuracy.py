import numpy as np
import scipy.stats

import ndop.measurements.data
from ndop.metos3d.model import Model

from util.debug import Debug

class Accuracy(Debug):
    
    def __init__(self, debug_level=0, required_debug_level=1):
        from ndop.metos3d.constants import MODEL_PARAMETER_DIM
        
        Debug.__init__(self, debug_level, required_debug_level-1, 'ndop.optimization.accuracy: ')
        
        self.print_debug_inc('Initiating accuracy object.')
        
        self.means = ndop.measurements.data.means(self.debug_level, self.required_debug_level + 1)
        self.mos = ndop.measurements.data.mos(self.debug_level, self.required_debug_level + 1)
        
        nobs = ndop.measurements.data.nobs(self.debug_level, self.required_debug_level + 1)
        varis = ndop.measurements.data.varis(self.debug_level, self.required_debug_level + 1)
        self.nobs = nobs
        self.varis = varis
        self.nobs_per_vari = nobs / varis
        p_dim = MODEL_PARAMETER_DIM
        
        axis_sum = tuple(range(1, len(nobs.shape)))
        self.averaged_model_variance_axis_sum = axis_sum
        
        self.averaged_model_variance_factors = 1 / (np.nansum(nobs, axis=axis_sum) - MODEL_PARAMETER_DIM)
        
        self._averaged_model_variance = np.nansum(nobs * varis, axis=axis_sum) / np.nansum(nobs, axis=axis_sum)
        
        self.print_debug_dec('Accuracy object initiated.')
    
    
#     @property
    def averaged_model_variance(self):
        return self._averaged_model_variance
    
    def averaged_model_variance_estimation(self, model_f):
        model_f_mos = model_f ** 2
        
        means = self.means
        mos = self.mos
        factors = self.averaged_model_variance_factors
        axis_sum = self.averaged_model_variance_axis_sum
        
        ave = factors * np.nansum(mos - 2 * means * model_f + model_f_mos, axis=axis_sum)
        
        return ave
    
    
    def covariance_for_parameters(self, model_df):
        self.print_debug_inc('Calculating covariance matrix.')
        
        df = model_df
        factors = self.nobs_per_vari
        
        df_shape = df.shape
        p_dim = df_shape[-1]
        
        matrix = np.zeros([p_dim, p_dim], dtype=np.float64)
        
        for multi_index in np.ndindex(*df_shape[:-1]):
            factor_i = factors[multi_index]
            if factor_i > 0:
                df_i = df[multi_index]
                matrix += (np.outer(df_i, df_i) * factor_i)
        
        matrix = np.linalg.inv(matrix)
        
        self.print_debug_dec('Covariance matrix calculated.')
        
        return matrix
    
    
    
    def confidence_for_parameters(self, model_df):
        self.print_debug_inc('Calculating confidence for parameters.')
        
        C = self.covariance_for_parameters(model_df)
        confidence = self.confidence(C)
        
        self.print_debug_dec('Confidence for parameters calculated.')
        
        return confidence
    
    
    
    def variance_for_model(self, covariance_for_parameters, df):
        self.print_debug_inc('Calculating variance for model.')
        
        df_shape = df.shape
        p_dim = df_shape[-1]
        
        C_p = covariance_for_parameters.view(type=np.matrix)
        matrix = np.empty(df_shape[:-1], dtype=np.float64)
        
        for i in np.ndindex(*matrix.shape):
            df_i = df[i]
            if not any(np.isnan(df_i)):
                df_i = df[i].view(type=np.matrix).T
                matrix[i] = df_i.T * C_p * df_i
            else:
                matrix[i] = np.nan
        
        self.print_debug_dec('Variance for model calculated.')
        
        return matrix
    
    
    def confidence_for_model(self, model_df):
        self.print_debug_inc('Calculating confidence for model.')
        
        C_p = self.covariance_for_parameters(model_df)
        C_m = self.variance_for_model(C_p, model_df)
        confidence_factor = self.confidence(np.array([1]))
        confidence = np.sqrt(C_m) * confidence_factor
        
        self.print_debug_dec('Confidence for model calculated.')
        
        return confidence
    
    
    
    
    def confidence(self, covariance, alpha = 0.99, debug_level = 0, required_debug_level = 1):
        C = covariance
        d = np.diag(C)
        
        n = C.shape[0]
        
        # calculate chi-square quantil with confidence level alpha and n degrees of freedom
        gamma = scipy.stats.chi2.ppf(alpha, n)
        
        confidence_factors = d**(1/2) * gamma**(1/2)
        
        return confidence_factors
    
#     def averaged_variance(self):
#         nobs = self.nobs
#         varis = self.varis
#         
#         averaged_variance = np.nansum(nobs * varis, axis=nobs.shape[1:]) / np.nansum(nobs, axis=nobs.shape[1:])
#         
#         return ave