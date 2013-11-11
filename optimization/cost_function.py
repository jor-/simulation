import numpy as np

import measurements.data
from ndop.metos3d.model import Model
from util.debug import Debug




class Cost_Function_Base(Debug):
    
    def __init__(self, years=10000, tolerance=0, time_step_size=1, debug_level=0, required_debug_level=1):
        from ndop.metos3d.constants import MODEL_PARAMETER_DIM
        
        Debug.__init__(self, debug_level, required_debug_level-1, 'ndop.optimization.cost_function: ')
        
        self.print_debug_inc('Initiating cost function.')
        
        self.years = years
        self.tolerance = tolerance
        self.time_step_size = time_step_size
        
        self.model = Model(self.debug_level, self.required_debug_level + 1)
        
        self.means = measurements.data.means(self.debug_level, self.required_debug_level + 1)
        
        nobs = measurements.data.nobs(self.debug_level, self.required_debug_level + 1)
        varis = measurements.data.varis(self.debug_level, self.required_debug_level + 1)
        self.nobs_per_vari = nobs / varis
        self.factor = 1 / ((nobs > 0).sum() - MODEL_PARAMETER_DIM)
        
        self.last_parameters = None
        
        self.print_debug_dec('Cost function initiated.')
    
    
    def model_f(self, parameters):
        if self.last_parameters is not None and all(parameters == self.last_parameters):
            model_f = self.last_model_f
        else:
            model_f = self.model.f(parameters, years=self.years, tolerance=self.tolerance, time_step_size=self.time_step_size)
            self.last_parameters = parameters
            self.last_model_f = model_f
        
        return model_f
    
    
    def model_df(self, parameters, accuracy_order=1):
        model_df = self.model.df(parameters, years=self.years, tolerance=self.tolerance, time_step_size=self.time_step_size, accuracy_order=accuracy_order)
        
        return model_df




class Cost_Function_1(Cost_Function_Base):
    
    def __init__(self, years=10000, tolerance=0, time_step_size=1, debug_level=0, required_debug_level=1):
        from ndop.metos3d.constants import MODEL_PARAMETER_DIM
        
        Cost_Function_Base.__init__(self, years, tolerance, time_step_size, debug_level, required_debug_level)
        
        self.means = measurements.data.means(self.debug_level, self.required_debug_level + 1)
        self.factor = 1 / ((nobs > 0).sum() - MODEL_PARAMETER_DIM)
    
    
    def f(self, parameters):
        model_f = self.model_f(parameters)
        factor = self.factor
        
        f = factor * np.nansum((means - model_f)**2)
        
        return f
    
    
    def df(self, parameters, accuracy_order=1):
        model_f = self.model_f(parameters)
        model_df = self.model_df(parameters, accuracy_order)
        
        means = self.means
        factor = self.factor
        
        df_factors = - 2 * factor * (means - model_f)
        
        p_dim = len(parameters)
        df = np.empty(p_dim)
        
        for i in range(p_dim):
            df[i] = np.nansum(df_factors * model_df[..., i])
        
        df = factor * df 
        
        return df




class Cost_Function_2(Cost_Function_Base):
    
    def __init__(self, years=10000, tolerance=0, time_step_size=1, debug_level=0, required_debug_level=1):
        from ndop.metos3d.constants import MODEL_PARAMETER_DIM
        
        Cost_Function_Base.__init__(self, years, tolerance, time_step_size, debug_level, required_debug_level)
        
        self.means = measurements.data.means(self.debug_level, self.required_debug_level + 1)
        
        nobs = measurements.data.nobs(self.debug_level, self.required_debug_level + 1)
        varis = measurements.data.varis(self.debug_level, self.required_debug_level + 1)
        self.nobs_per_vari = nobs / varis
        self.factor = 1 / ((nobs > 0).sum() - MODEL_PARAMETER_DIM)
    
    
    def f(self, parameters):
        model_f = self.model_f(parameters)
        
        means = self.means
        nobs_per_vari = self.nobs_per_vari
        factor = self.factor
        
        f = factor * np.nansum(nobs_per_vari * (means - model_f)**2)
        
        return f
    
    
    def df(self, parameters, accuracy_order=1):
        model_f = self.model_f(parameters)
        model_df = self.model_df(parameters, accuracy_order=accuracy_order)
        
        means = self.means
        nobs_per_vari = self.nobs_per_vari
        factor = self.factor
        
        df_factors = - 2 * factor * nobs_per_vari * (means - model_f)
        
        p_dim = len(parameters)
        df = np.empty(p_dim)
        
        for i in range(p_dim):
            df[i] = np.nansum(df_factors * model_df[..., i])
        
        return df



# class Cost_Function_2(Debug):
#     
#     def __init__(self, years=10000, tolerance=0, time_step_size=1, debug_level=0, required_debug_level=1):
#         from ndop.metos3d.constants import MODEL_PARAMETER_DIM
#         
#         Debug.__init__(self, debug_level, required_debug_level-1, 'ndop.optimization.cost_function: ')
#         
#         self.print_debug_inc('Initiating cost function.')
#         
#         self.years = years
#         self.tolerance = tolerance
#         self.time_step_size = time_step_size
#         
#         self.model = Model(self.debug_level, self.required_debug_level + 1)
#         
#         self.means = measurements.data.means(self.debug_level, self.required_debug_level + 1)
#         
#         nobs = measurements.data.nobs(self.debug_level, self.required_debug_level + 1)
#         varis = measurements.data.varis(self.debug_level, self.required_debug_level + 1)
#         self.nobs_per_vari = nobs / varis
#         self.factor = 1 / ((nobs > 0).sum() - MODEL_PARAMETER_DIM)
#         
#         self.last_parameters = None
#         
#         self.print_debug_dec('Cost function initiated.')
#     
#     
#     
#     def f(self, parameters):
#         if self.last_parameters is not None and all(parameters == self.last_parameters):
#             model_f = self.last_model_f
#         else:
#             model_f = self.model.f(parameters, years=self.years, tolerance=self.tolerance, time_step_size=self.time_step_size)
#             self.last_parameters = parameters
#             self.last_model_f = model_f
#         
#         means = self.means
#         nobs_per_vari = self.nobs_per_vari
#         factor = self.factor
#         
#         f = factor * np.nansum(nobs_per_vari * (means - model_f)**2)
#         
#         return f
#     
#     
#     def df(self, parameters, accuracy_order=1):
#         model_df = self.model.df(parameters, years=self.years, tolerance=self.tolerance, time_step_size=self.time_step_size, accuracy_order=accuracy_order)
#         
#         if self.last_parameters is not None and all(parameters == self.last_parameters):
#             model_f = self.last_model_f
#         else:
#             model_f = self.model.f(parameters, years=self.years, tolerance=self.tolerance, time_step_size=self.time_step_size)
#             self.last_parameters = parameters
#             self.last_model_f = model_f
#         
#         means = self.means
#         nobs_per_vari = self.nobs_per_vari
#         factor = self.factor
#         
#         df_factors = - 2 * nobs_per_vari * (means - model_f)
#         
#         p_dim = len(parameters)
#         df = np.empty(p_dim)
#         
#         for i in range(p_dim):
#             df[i] = np.nansum(df_factors * model_df[..., i])
#         
#         df = factor * df 
#         
#         return df