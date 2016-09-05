import numpy as np

import util.options
import util.cache
import util.logging

import simulation.model.constants

logger = util.logging.logger



class ModelOptions(util.options.Options):
    
    OPTIONS = ('model_name', 'time_step', 'parameters', 'spinup_options', 'derivative_options', 'parameter_tolerance_options', 'initial_concentration_options')

    def __init__(self, options=None):
        
        default_parameter_tolerance_options = ParameterToleranceOptions(options={'relative': np.array([0]), 'absolute': np.array([10**(- simulation.model.constants.DATABASE_PARAMETERS_RELIABLE_DECIMAL_PLACES)])})
        default_options = {'model_name': simulation.model.constants.MODEL_NAMES[0], 'time_step': 1, 'spinup_options': SpinupOptions(), 'derivative_options': DerivativeOptions(), 'parameter_tolerance_options': default_parameter_tolerance_options, 'initial_concentration_options': InitialConcentrationOptions()}

        super().__init__(options=options, default_options=default_options, option_names=ModelOptions.OPTIONS)
        
        self.add_dependency('model_name', 'parameters' , dependent_option_object=self)
        if 'model_name' in self and 'parameters' in self:
            self.parameters_check(self['parameters'])
    
    
    ## options
    
    def model_name_check(self, model_name):
        if not model_name in simulation.model.constants.MODEL_NAMES:
            raise ValueError('Model name {} is unknown. Only the names {} are supported.'.format(model_name, simulation.model.constants.MODEL_NAMES))
    

    def time_step_check(self, time_step):
        if not time_step in simulation.model.constants.METOS_TIME_STEPS:
            raise ValueError('Wrong time_step in model options. Time step has to be in {} .'.format(time_step, simulation.model.constants.METOS_TIME_STEPS))
        assert simulation.model.constants.METOS_T_DIM % time_step == 0
        

    def parameters_check(self, parameters):
        parameters = np.asanyarray(parameters)
        
        ## check if matching to model
        if 'model_name' in self:
            
            ## check length
            if len(parameters) != self.parameters_len:
                raise ValueError('The model parameters {} are not allowed. The length of the model parameters have to be {} but it is {}.'.format(parameters, self.parameters_len, len(parameters)))
            
            ## check bounds
            if any(parameters < self.parameters_lower_bounds):
                indices = np.where(parameters < self.parameters_lower_bounds)
                raise ValueError('The model parameters {} are not allowed. The model parameters with the indices {} are below their lower bound {}.'.format(parameters, indices, self.parameters_lower_bounds[indices]))
    
            if any(parameters > self.parameters_upper_bounds):
                indices = np.where(parameters > self.parameters_upper_bounds)
                raise ValueError('The model parameters {} are not allowed. The model parameters with the indices {} are above their upper bound {}.'.format(parameters, indices, self.parameters_upper_bounds[indices]))
        
        return tuple(parameters)
    
    
    def parameters_changed(self, independent_option, new_value):
        assert independent_option == 'model_name'

        try:
            parameters = self._options['parameters']
        except KeyError:
            remove = False
        else:
            try:
                self.parameters_check(parameters)
            except ValueError:
                remove = True
            else:
                remove = False
        return remove
    

    def spinup_options_check(self, spinup_options):
        return util.options.as_options(spinup_options, SpinupOptions)
    

    def derivative_options_check(self, derivative_options):
        derivative_options = util.options.as_options(derivative_options, DerivativeOptions)
        derivative_options._model_options = self
        return derivative_options
    

    def parameter_tolerance_options_check(self, parameter_tolerance_options):
        parameter_tolerance_options = util.options.as_options(parameter_tolerance_options, ParameterToleranceOptions)
        parameter_tolerance_options._model_options = self
        return parameter_tolerance_options


    def initial_concentration_options_check(self, initial_concentration_options):
        initial_concentration_options = util.options.as_options(initial_concentration_options, InitialConcentrationOptions)
        initial_concentration_options._model_options = self
        return initial_concentration_options
    
    
    ## properties
    
    @property
    def tracers(self):
        return simulation.model.constants.MODEL_TRACER[self.model_name]
    
    @property
    def tracers_len(self):
        return len(self.tracers)
    
    @property
    def time_steps_per_year(self):
        return int(simulation.model.constants.METOS_T_DIM / self.time_step)

    @property
    def parameters_bounds(self):
        return simulation.model.constants.MODEL_PARAMETER_BOUNDS[self.model_name]
    
    @property
    def parameters_lower_bounds(self):
        return self.parameters_bounds[:,0]
    
    @property
    def parameters_upper_bounds(self):
        return self.parameters_bounds[:,1]

    @property
    def parameters_len(self):
        return len(self.parameters_bounds)
    



class SpinupOptions(util.options.Options):
    
    OPTIONS = ('years', 'tolerance', 'combination')

    def __init__(self, options=None):
        super().__init__(options=options, default_options=simulation.model.constants.MODEL_DEFAULT_SPINUP_OPTIONS, option_names=SpinupOptions.OPTIONS)
    
    
    ## options methods

    def years_check(self, years):
        if years < 0:
            raise ValueError('Years must be greater or equal to 0, but it is {} .'.format(years))
    

    def tolerance_check(self, tolerance):
        if tolerance < 0:
            raise ValueError('Tolerance must be greater or equal to 0, but it is {} .'.format(tolerance))
    

    def combination_check(self, combination):
        POSSIBLE_VALUES = ['and', 'or']
        if combination not in POSSIBLE_VALUES:
            raise ValueError('Combination "{}" unknown. Possible combinations are: {}'.format(combination, POSSIBLE_VALUES))




class DerivativeOptions(util.options.Options):
    
    OPTIONS = ('years', 'step_size', 'accuracy_order')

    def __init__(self, options=None):
        super().__init__(options=options, default_options=simulation.model.constants.MODEL_DEFAULT_DERIVATIVE_OPTIONS, option_names=DerivativeOptions.OPTIONS)
        self.__model_options = None

    
    ## options

    def years_check(self, years):
        if years < 0:
            raise ValueError('Years must be greater or equal to 0, but it is {} .'.format(years))
    

    def step_size_check(self, step_size):
        if step_size < 0:
            raise ValueError('Step size must be greater or equal to 0, but it is {} .'.format(step_size))
    

    def accuracy_order_check(self, accuracy_order):
        POSSIBLE_VALUES = [1, 2]
        if accuracy_order not in POSSIBLE_VALUES:
            raise ValueError('Accuracy_order "{}" unknown. Possible accuracy_orders are: {}'.format(accuracy_order, POSSIBLE_VALUES))


    ## properties
    
    @property
    def _model_options(self):
        return self.__model_options
    
    @_model_options.setter
    def _model_options(self, value):
        self.__model_options = value
    
    @property
    def _model_name(self):
        return self._model_options.model_name
        
    
    @property
    def parameters_typical_values(self):
        return simulation.model.constants.MODEL_PARAMETER_TYPICAL[self._model_name]




class ToleranceOptions(util.options.Options):
    
    OPTIONS = ('relative', 'absolute')

    def __init__(self, options=None):
        default_options={'relative': np.array([0]), 'absolute': np.array([0])}
        super().__init__(options=options, default_options=default_options, option_names=ToleranceOptions.OPTIONS)
    

    ## options
    
    @property
    def _values_len(self):
        return None
    
    
    def _check_tolerance(self, tolerance):
        tolerance = np.asanyarray(tolerance)
        if tolerance.ndim == 0:
            tolerance = tolerance.reshape(-1)
        
        if tolerance.ndim != 1:
            raise ValueError('The tolerances must be a scalar or a vector, but the tolerance is {}.'.format(tolerance))
        
        if len(tolerance) != 1:
            if self._values_len is None:
                raise ValueError('The tolerances must be a scalar, but its length is {}.'.format(len(tolerance)))
            elif self._values_len != len(tolerance):
                raise ValueError('The tolerances must be a scalar or a vector with length {}, but its length is {}.'.format(self._value_len, len(tolerance)))
        
        if np.any(tolerance < 0):
            raise ValueError('Tolerance must be greater or equal to 0 in all components, but it is {} .'.format(tolerance))
        
        if len(tolerance):
            tolerance = tolerance[0]
        else:
            tolerance = tuple(tolerance)
        
        return tolerance
    
    
    def relative_check(self, relative):
        return self._check_tolerance(relative)


    def absolute_check(self, absolute):
        return self._check_tolerance(absolute)
    
    
    def _check_value_changed(self, dependent_option, independent_option, new_value):
        assert independent_option == 'model_name'

        ## check if option is set
        try:
            value =  self._options[dependent_option]
        except KeyError:
            remove = False
        ## check len of option value if sequence
        else:
            try:
                len_value = len(value)
            except TypeError:
                remove = False
            else:
                remove = len_value != self._values_len
        ## set 0 tolerance instead of remove
        if remove:
            self._set_option(dependent_option, 0)
            remove = False
        
        return remove
    
    
    def relative_depending_value_changed(self, independent_option, new_value):
        self._check_value_changed('relative', independent_option, new_value)

    
    def absolute_depending_value_changed(self, independent_option, new_value):
        self._check_value_changed('absolute', independent_option, new_value)




class ParameterToleranceOptions(ToleranceOptions):
    
    @property
    def _model_options(self):
        return self.__model_options
    
    @_model_options.setter
    def _model_options(self, model_options):
        self.__model_options = model_options
        model_options.add_dependency('model_name', 'relative' , dependent_option_object=self)
        model_options.add_dependency('model_name', 'absolute' , dependent_option_object=self)
        if 'model_name' in model_options:
            if 'relative' in self:
                self.relative_check(self['relative'])
            if 'absolute' in self:
                self.absolute_check(self['absolute'])
    
    @property
    def _values_len(self):
        try:
            model_options = self._model_options
        except AttributeError:
            return None
        else:
            return model_options._parameters_len




class InitialConcentrationOptions(util.options.Options):
    
    OPTIONS = ('concentrations', 'tolerance_options')

    def __init__(self, options=None):
        
        
        default_tolerance_options = ToleranceOptions(options={'relative': np.array([0]), 'absolute': np.array([10**(- simulation.model.constants.DATABASE_CONSTANT_CONCENTRATIONS_RELIABLE_DECIMAL_PLACES)])})
        default_options = {'tolerance_options': default_tolerance_options}
        
        super().__init__(options=options, default_options=default_options, option_names=InitialConcentrationOptions.OPTIONS)
        
        self.__model_options = None

    
    ## options
    
    @property
    def _model_options(self):
        return self.__model_options
    
    @_model_options.setter
    def _model_options(self, model_options):
        self.__model_options = model_options
        model_options.add_dependency('model_name', 'concentrations' , dependent_option_object=self)
        if 'model_name' in model_options and 'concentrations'in self:
            self.concentrations_check(self['concentrations'])
    
    
    @property
    def _model_name(self):
        try:
            model_options = self._model_options
        except AttributeError:
            return None
        else:
            return model_options.model_name

    
    @property
    def _tracers_len(self):
        try:
            model_options = self._model_options
        except AttributeError:
            return None
        else:
            return model_options.tracers_len
    
    
    def concentrations_check(self, concentrations):
        ## check concentration input
        if self._tracers_len is not None and len(concentrations) != self._tracers_len:
            raise ValueError('The concentrations must be an iterable with length {}, but its length is {}.'.format(self._tracers_len, len(concentrations)))
        
        concentrations = np.asanyarray(concentrations)
        
        if np.any(concentrations < 0):
            raise ValueError('Concentrations must be greater or equal to 0 in all components, but it is {} .'.format(concentrations))
        
        if concentrations.ndim == 0:
            concentrations = concentrations.reshape(-1)
        
        if concentrations.ndim not in (1, 2):
            raise ValueError('The concentration must have 1 or 2 dimensions, but it has {} dimensions.'.format(concentrations.ndim))
        
        if concentrations.ndim == 2 and concentrations.shape[1] != simulation.model.constants.METOS_VECTOR_LEN:
            raise ValueError('The concentrations with 2 dimensions, must have {} as second dimensions, but it is {}.'.format(simulation.model.constants.METOS_VECTOR_LEN, concentrations.shape[1]))
        
        ## return as tuple
        if concentrations.ndim == 2:
            return tuple(map(tuple, concentrations))
        else:
            return tuple(concentrations)
    
    
    def concentrations_depending_value_changed(self, independent_option, new_value):
        assert independent_option == 'model_name'
        try:
            concentrations = self._options['concentrations']
        except KeyError:
            remove = False
        else:
            remove = simulation.model.constants.MODEL_TRACER[self._model_name] != simulation.model.constants.MODEL_TRACER[new_value]
        return remove
    
    
    @property
    def concentrations_calculated(self):
        if self._model_name is not None:
            return simulation.model.constants.MODEL_DEFAULT_INITIAL_CONCENTRATION[self._model_name]
        else:
            return None
    

    def tolerance_options_check(self, tolerance_options):
        tolerance_options = util.options.as_options(tolerance_options, ToleranceOptions)
        return tolerance_options
    
    
    ## properties
    
    @property
    def use_constant_concentrations(self):
        concentration = self.concentrations[0]
        try:
            len(concentration)
        except TypeError:
            return True
        else:
            return False
    
    
    @property
    def concentration_len(self):
        if self.use_constant_concentrations:
            return 1
        else:
            return len(concentrations[0])



def as_model_options(model_options):
    if not isinstance(model_options, ModelOptions):
        model_options = ModelOptions(model_options)
    return model_options
