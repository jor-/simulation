import numpy as np

import ndop.measurements.data
from ndop.metos3d.model import Model
from util.debug import Debug

class Cost_function(Debug):
    
    def __init__(self, years=7000, tolerance=0, time_step_size=1, debug_level=0, required_debug_level=1):
        Debug.__init__(self, debug_level, required_debug_level-1, 'ndop.optimization.cost_function: ')
        
        self.print_debug_inc('Initiating cost function.')
        
        self.years = years
        self.tolerance = tolerance
        self.time_step_size = time_step_size
        
        self.model = Model(self.debug_level, self.required_debug_level + 1)
        
        self.means = ndop.measurements.data.means(self.debug_level, self.required_debug_level + 1)
        self.mos = ndop.measurements.data.mos(self.debug_level, self.required_debug_level + 1)
        
        nobs = ndop.measurements.data.nobs(self.debug_level, self.required_debug_level + 1)
        varis = ndop.measurements.data.varis(self.debug_level, self.required_debug_level + 1)
        self.nobs_per_vari = nobs / varis
        self.factor = 1 / np.nansum(nobs)
        
        self.last_parameters = None
        
        self.print_debug_dec('Cost function initiated.')
    
    
    
    def f(self, parameters):
        if self.last_parameters is not None and all(parameters == self.last_parameters):
            model_f = self.last_model_f
        else:
            model_f = self.model.f(parameters, years=self.years, tolerance=self.tolerance, time_step_size=self.time_step_size)
            self.last_parameters = parameters
            self.last_model_f = model_f
        
        model_f_mos = model_f ** 2
        
        means = self.means
        mos = self.mos
        nobs_per_vari = self.nobs_per_vari
        factor = self.factor
        
        f = factor * np.nansum(nobs_per_vari * (mos - 2 * means * model_f + model_f_mos))
        
        return f
    
    
    def df(self, parameters, accuracy_order=1):
        model_df = self.model.df(parameters, years=self.years, tolerance=self.tolerance, time_step_size=self.time_step_size, accuracy_order=accuracy_order)
        
        if self.last_parameters is not None and all(parameters == self.last_parameters):
            model_f = self.last_model_f
        else:
            model_f = self.model.f(parameters, years=self.years, tolerance=self.tolerance, time_step_size=self.time_step_size)
            self.last_parameters = parameters
            self.last_model_f = model_f
        
        means = self.means
        nobs_per_vari = self.nobs_per_vari
        factor = self.factor
        
        df_factors = 2 * nobs_per_vari * (model_f - means)
        
        p_dim = len(parameters)
        df = np.empty(p_dim)
        
        for i in range(p_dim):
            df[i] = np.nansum(df_factors * model_df[..., i])
        
        df = factor * df 
        
        return df