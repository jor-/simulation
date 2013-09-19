import os.path

import ndop.metos3d.direct_access
from ndop.analysis.accuracy import Accuracy

import util.cache
from util.debug import Debug

class Accuracy_Cached(Debug):
    
    def __init__(self, debug_level=0, required_debug_level=1):
        Debug.__init__(self, debug_level, required_debug_level-1, 'ndop.optimization.accuracy_cached: ')
        
        accuracy = Accuracy(debug_level, required_debug_level+1)
        
        self.accuracy = accuracy
        self.means = accuracy.means
        self.nobs = accuracy.nobs
        self.varis = accuracy.varis
        self.nobs_per_vari = accuracy.nobs_per_vari
        self.vari_of_means = accuracy.vari_of_means
        self.number_of_not_empty_boxes = accuracy.number_of_not_empty_boxes
        self.averaged_model_variance = accuracy.averaged_model_variance
    
    
    def calculate_confidence_for_parameters(self, parameter_set_dir):
        df = ndop.metos3d.direct_access.get_df(parameter_set_dir, debug_level=self.debug_level, required_debug_level=self.required_debug_level+1)
        confidence = self.accuracy.confidence_for_parameters(df)
        
        return confidence
        
    
    def confidence_for_parameters(self, parameter_set_dir):
        from ndop.analysis.constants import ANALYSIS_OUTPUT_DIRNAME, CONFIDENCE_FOR_PARAMETERS_FILENAME
        
        confidence_file = os.path.join(parameter_set_dir, ANALYSIS_OUTPUT_DIRNAME, CONFIDENCE_FOR_PARAMETERS_FILENAME)
        calculate_method = lambda : self.calculate_confidence_for_parameters(parameter_set_dir)
        
        confidence = util.cache.load_or_calculate(confidence_file, calculate_method, debug_level=self.debug_level, required_debug_level=self.required_debug_level+1)
        
        return confidence
    
    
    
    def calculate_confidence_for_model(self, parameter_set_dir):
        df = ndop.metos3d.direct_access.get_df(parameter_set_dir, debug_level=self.debug_level, required_debug_level=self.required_debug_level+1)
        confidence = self.accuracy.confidence_for_model(df)
        
        return confidence
    
    
    def confidence_for_model(self, parameter_set_dir):
        from ndop.analysis.constants import ANALYSIS_OUTPUT_DIRNAME, CONFIDENCE_FOR_MODEL_FILENAME
        
        confidence_file = os.path.join(parameter_set_dir, ANALYSIS_OUTPUT_DIRNAME, CONFIDENCE_FOR_MODEL_FILENAME)
        calculate_method = lambda : self.calculate_confidence_for_model(parameter_set_dir)
        
        confidence = util.cache.load_or_calculate(confidence_file, calculate_method, debug_level=self.debug_level, required_debug_level=self.required_debug_level+1)
        
        return confidence
    
    
    
    def calculate_averaged_model_variance(self):
        return self.accuracy.averaged_model_variance()
    
    
    def averaged_model_variance(self, parameter_set_dir):
        from ndop.analysis.constants import ANALYSIS_OUTPUT_DIRNAME, AVERAGED_MODEL_VARIANCE_FILENAME
        
        variance_file = os.path.join(parameter_set_dir, ANALYSIS_OUTPUT_DIRNAME, AVERAGED_MODEL_VARIANCE_FILENAME)
        calculate_method = lambda : self.calculate_averaged_model_variance()
        
        variance = util.cache.load_or_calculate(variance_file, calculate_method, debug_level=self.debug_level, required_debug_level=self.required_debug_level+1)
        
        return variance
    
    
    
    def calculate_averaged_model_variance_estimation(self, parameter_set_dir):
        f = ndop.metos3d.direct_access.get_f(parameter_set_dir, debug_level=self.debug_level, required_debug_level=self.required_debug_level+1)
        
        variance_estimation = self.accuracy.averaged_model_variance_estimation(f)
        
        return variance_estimation
    
    
    def averaged_model_variance_estimation(self, parameter_set_dir):
        from ndop.analysis.constants import ANALYSIS_OUTPUT_DIRNAME, AVERAGED_MODEL_VARIANCE_ESTIMATION_FILENAME
        
        variance_file = os.path.join(parameter_set_dir, ANALYSIS_OUTPUT_DIRNAME, AVERAGED_MODEL_VARIANCE_ESTIMATION_FILENAME)
        calculate_method = lambda : self.calculate_averaged_model_variance_estimation(parameter_set_dir)
        
        variance = util.cache.load_or_calculate(variance_file, calculate_method, debug_level=self.debug_level, required_debug_level=self.required_debug_level+1)
        
        return variance
    
    
    
    def calculate_probability_of_observations(self, parameter_set_dir):
        f = ndop.metos3d.direct_access.get_f(parameter_set_dir, debug_level=self.debug_level, required_debug_level=self.required_debug_level+1)
        
        probability = self.accuracy.probability_of_observations(f)
        
        return probability
    
    
    def probability_of_observations(self, parameter_set_dir):
        from ndop.analysis.constants import ANALYSIS_OUTPUT_DIRNAME, OBSERVATION_PROPERTY_FILENAME
        
        file = os.path.join(parameter_set_dir, ANALYSIS_OUTPUT_DIRNAME, OBSERVATION_PROPERTY_FILENAME)
        calculate_method = lambda : self.calculate_probability_of_observations(parameter_set_dir)
        
        values = util.cache.load_or_calculate(file, calculate_method, debug_level=self.debug_level, required_debug_level=self.required_debug_level+1)
        
        return values
    
    
    
    def calculate_averaged_probability_of_observations(self, parameter_set_dir):
        f = ndop.metos3d.direct_access.get_f(parameter_set_dir, debug_level=self.debug_level, required_debug_level=self.required_debug_level+1)
        
        probability = self.accuracy.averaged_probability_of_observations(f)
        
        return probability
    
    
    def averaged_probability_of_observations(self, parameter_set_dir):
        from ndop.analysis.constants import ANALYSIS_OUTPUT_DIRNAME, AVERAGED_OBSERVATION_PROPERTY_FILENAME
        
        file = os.path.join(parameter_set_dir, ANALYSIS_OUTPUT_DIRNAME, AVERAGED_OBSERVATION_PROPERTY_FILENAME)
        calculate_method = lambda : self.calculate_averaged_probability_of_observations(parameter_set_dir)
        
        values = util.cache.load_or_calculate(file, calculate_method, debug_level=self.debug_level, required_debug_level=self.required_debug_level+1)
        
        return values