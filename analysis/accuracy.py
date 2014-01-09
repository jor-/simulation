import numpy as np
import scipy.stats

import measurements.po4.woa.data
from ndop.metos3d.model import Model

from util.debug import Debug

class Accuracy(Debug):
    
    def __init__(self, debug_level=0, required_debug_level=1):
        from ndop.metos3d.constants import MODEL_PARAMETER_DIM
        
        Debug.__init__(self, debug_level, required_debug_level-1, 'ndop.optimization.accuracy: ')
        
        self.print_debug_inc('Initiating accuracy object.')
        
        self.means = measurements.po4.woa.data.means(self.debug_level, self.required_debug_level + 1)
        
        nobs = measurements.po4.woa.data.nobs(self.debug_level, self.required_debug_level + 1)
        varis = measurements.po4.woa.data.varis(self.debug_level, self.required_debug_level + 1)
        self.nobs = nobs
        self.varis = varis
        self.nobs_per_vari = nobs / varis
        
        nobs_with_nans = np.copy(nobs)
        nobs_with_nans[nobs_with_nans == 0] = np.nan 
        self.vari_of_means = varis / nobs_with_nans
        
        axis_sum = tuple(range(1, len(nobs.shape)))
        
        number_of_not_empty_boxes = np.sum((nobs > 0), axis=axis_sum)
        self.number_of_not_empty_boxes = number_of_not_empty_boxes
        
        ## calculate averaged measurement variance
        model_out_dim = nobs.shape[0]
        averaged_measurement_variance = np.empty(model_out_dim, dtype=np.float64)
        
        for i in range(model_out_dim):
            varis_i = varis[i]
            nobs_i = nobs[i]
            number_of_not_empty_boxes_i = number_of_not_empty_boxes[i]
            
            averaged_measurement_variance[i] = np.sum(varis_i[nobs_i > 0] / nobs_i[nobs_i > 0]) / number_of_not_empty_boxes_i
            
        self._averaged_measurement_variance = averaged_measurement_variance
        
        self.print_debug_dec('Accuracy object initiated.')
    
    
#     @property
    def averaged_measurement_variance(self):
        return self._averaged_measurement_variance
    
    def averaged_measurement_variance_estimated_with_model(self, model_f):
        from ndop.metos3d.constants import MODEL_PARAMETER_DIM
        
        means = self.means
        number_of_not_empty_boxes = self.number_of_not_empty_boxes
        
        axis_sum = tuple(range(1, len(model_f.shape)))
        factors = 1 / (number_of_not_empty_boxes - MODEL_PARAMETER_DIM)
        
        ave = factors * np.nansum((means - model_f)**2, axis=axis_sum)
        
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
    
    
    
    def probability_of_observations(self, model_f):
        self.print_debug_inc('Calculating probability of observations.')
        
        means = self.means
        vari_of_means = self.vari_of_means
        
        probability = scipy.stats.norm.logpdf(np.abs(means - model_f), scale=(vari_of_means))
        
        self.print_debug_dec('Probability of observations calculated.')
        
        return probability
    
    
    def averaged_probability_of_observations(self, model_f):
        probability = self.probability_of_observations(model_f)
        number_of_not_empty_boxes = self.number_of_not_empty_boxes
        
        axis_sum = tuple(range(1, len(model_f.shape)))
        average = np.nansum(probability, axis=axis_sum) / number_of_not_empty_boxes
        
        return average