import os.path

import ndop.metos3d.direct_access
from ndop.analysis.accuracy import Accuracy

import util.cache

class Accuracy_Cached(Accuracy):
    
    def __init__(self, debug_level=0, required_debug_level=1):
        super(Accuracy_Cached, self).__init__(debug_level, required_debug_level)
    
    
    def calculate_confidence_for_parameters(self, parameter_set_dir):
        df = ndop.metos3d.direct_access.get_df(parameter_set_dir, debug_level=self.debug_level+1, required_debug_level=self.required_debug_level)
        confidence = super(Accuracy_Cached, self).confidence_for_parameters(df)
        
        return confidence
        
    
    def confidence_for_parameters(self, parameter_set_dir):
        from ndop.analysis.constants import ANALYSIS_OUTPUT_DIRNAME, CONFIDENCE_FOR_PARAMETERS_FILENAME
        
        confidence_file = os.path.join(parameter_set_dir, ANALYSIS_OUTPUT_DIRNAME, CONFIDENCE_FOR_PARAMETERS_FILENAME)
        calculate_method = lambda : self.calculate_confidence_for_parameters(parameter_set_dir)
        
        confidence = util.cache.load_or_calculate(confidence_file, calculate_method, debug_level=self.debug_level+1, required_debug_level=self.required_debug_level)
        
        return confidence
    
    
    
    def calculate_confidence_for_model(self, parameter_set_dir):
        df = ndop.metos3d.direct_access.get_df(parameter_set_dir, debug_level=self.debug_level+1, required_debug_level=self.required_debug_level)
        confidence = super(Accuracy_Cached, self).confidence_for_model(df)
        
        return confidence
    
    
    def confidence_for_model(self, parameter_set_dir):
        from ndop.analysis.constants import ANALYSIS_OUTPUT_DIRNAME, CONFIDENCE_FOR_MODEL_FILENAME
        
        confidence_file = os.path.join(parameter_set_dir, ANALYSIS_OUTPUT_DIRNAME, CONFIDENCE_FOR_MODEL_FILENAME)
        calculate_method = lambda : self.calculate_confidence_for_model(parameter_set_dir)
        
        confidence = util.cache.load_or_calculate(confidence_file, calculate_method, debug_level=self.debug_level+1, required_debug_level=self.required_debug_level)
        
        return confidence
    
    
    
    def calculate_averaged_model_variance(self):
        return super(Accuracy_Cached, self).averaged_model_variance()
    
    
    def averaged_model_variance(self, parameter_set_dir):
        from ndop.analysis.constants import ANALYSIS_OUTPUT_DIRNAME, AVERAGED_MODEL_VARIANCE_FILENAME
        
        variance_file = os.path.join(parameter_set_dir, ANALYSIS_OUTPUT_DIRNAME, AVERAGED_MODEL_VARIANCE_FILENAME)
        calculate_method = lambda : self.calculate_averaged_model_variance()
        
        variance = util.cache.load_or_calculate(variance_file, calculate_method, debug_level=self.debug_level+1, required_debug_level=self.required_debug_level)
        
        return variance
    
    
    
    def calculate_averaged_model_variance_estimation(self, parameter_set_dir):
        f = ndop.metos3d.direct_access.get_f(parameter_set_dir, debug_level=self.debug_level+1, required_debug_level=self.required_debug_level)
        
        variance_estimation = super(Accuracy_Cached, self).averaged_model_variance_estimation(f)
        
        return variance_estimation
    
    
    def averaged_model_variance_estimation(self, parameter_set_dir):
        from ndop.analysis.constants import ANALYSIS_OUTPUT_DIRNAME, AVERAGED_MODEL_VARIANCE_ESTIMATION_FILENAME
        
        variance_file = os.path.join(parameter_set_dir, ANALYSIS_OUTPUT_DIRNAME, AVERAGED_MODEL_VARIANCE_ESTIMATION_FILENAME)
        calculate_method = lambda : self.calculate_averaged_model_variance_estimation(parameter_set_dir)
        
        variance = util.cache.load_or_calculate(variance_file, calculate_method, debug_level=self.debug_level+1, required_debug_level=self.required_debug_level)
        
        return variance