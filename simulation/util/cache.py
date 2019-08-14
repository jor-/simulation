import os.path

import numpy as np

import util.cache.memory
import util.parallel.with_multiprocessing

import measurements.universal.data

import simulation.model.cache
import simulation.model.constants


# Base

class Cache():

    def __init__(self, base_dir, measurements_object, model_options=None, model_job_options=None, include_initial_concentrations_factor_to_model_parameters=True):
        # set base dir
        self.base_dir = base_dir

        # set measurements
        self.measurements = measurements_object

        # prepare job options
        if model_job_options is None:
            model_job_options = {}
        try:
            model_job_options['name']
        except KeyError:
            model_job_options['name'] = str(self)

        try:
            model_job_options['nodes_setup']
        except KeyError:
            try:
                model_job_options['spinup']
            except KeyError:
                model_job_options['spinup'] = {}
            try:
                model_job_options['spinup']['nodes_setup']
            except KeyError:
                try:
                    model_job_options['spinup']['nodes_setup'] = simulation.model.constants.NODES_SETUP_SPINUP.copy()
                except AttributeError:
                    pass
            try:
                model_job_options['derivative']
            except KeyError:
                model_job_options['derivative'] = {}
            try:
                model_job_options['derivative']['nodes_setup']
            except KeyError:
                try:
                    model_job_options['derivative']['nodes_setup'] = simulation.model.constants.NODES_SETUP_DERIVATIVE.copy()
                except AttributeError:
                    pass
            try:
                model_job_options['trajectory']
            except KeyError:
                model_job_options['trajectory'] = {}
            try:
                model_job_options['trajectory']['nodes_setup']
            except KeyError:
                try:
                    model_job_options['trajectory']['nodes_setup'] = simulation.model.constants.NODES_SETUP_TRAJECTORY.copy()
                except AttributeError:
                    pass

        # set model
        self.model = simulation.model.cache.Model(model_options=model_options, job_options=model_job_options)

        # include_initial_concentrations_factor_to_model_parameters
        self.include_initial_concentrations_factor_to_model_parameters = include_initial_concentrations_factor_to_model_parameters

    # *** auxiliary getter and setter *** #

    @property
    def measurements(self):
        return self._measurements

    @measurements.setter
    def measurements(self, measurements_object):
        measurements_object = measurements.universal.data.as_measurements_collection(measurements_object)
        self._measurements = measurements_object

    @property
    def model_options(self):
        return self.model.model_options

    @model_options.setter
    def model_options(self, model_options):
        self.model.model_options = model_options

    @property
    def model_parameters(self):
        model_parameters = self.model_options.parameters
        if self.include_initial_concentrations_factor_to_model_parameters:
            model_name = self.model_options.model_name
            initial_concentrations = np.asarray(self.model_options.initial_concentration_options.concentrations)
            initial_base_concentrations = np.asarray(simulation.model.constants.MODEL_DEFAULT_INITIAL_CONCENTRATION[model_name])
            concentration_parameter = np.mean(initial_concentrations / initial_base_concentrations)
            model_parameters = np.array((*model_parameters, concentration_parameter))
        return model_parameters

    @model_parameters.setter
    def model_parameters(self, model_parameters):
        model_parameters_len = self.model_options.parameters_len
        model_parameters = np.asanyarray(model_parameters)
        if len(model_parameters) == model_parameters_len:
            self.model_options.parameters = model_parameters
            self.include_initial_concentrations_factor_to_model_parameters = False
        elif len(model_parameters) == model_parameters_len + 1:
            self.model_options.parameters = model_parameters[:-1]
            model_name = self.model_options.model_name
            initial_base_concentrations = np.asarray(simulation.model.constants.MODEL_DEFAULT_INITIAL_CONCENTRATION[model_name])
            self.model_options.initial_concentration_options.concentrations = initial_base_concentrations * model_parameters[-1]
            self.include_initial_concentrations_factor_to_model_parameters = True
        else:
            raise ValueError('The parameters for the model {} must be a vector of length {} or {}, but its length is {}.'.format(self.model_options.model_name, model_parameters_len, model_parameters_len + 1, len(model_parameters)))

    @property
    def model_parameters_len(self):
        model_parameters_len = self.model_options.parameters_len
        if self.include_initial_concentrations_factor_to_model_parameters:
            model_parameters_len += 1
        return model_parameters_len

    # *** names *** #

    @property
    def name(self):
        return self.__class__.__name__

    def __str__(self):
        return f'{self.name}({str(self.measurements)})'

    # *** cache *** #

    def _filename(self, filename):
        file = os.path.join(self.base_dir,
                            simulation.model.constants.DATABASE_CACHE_SPINUP_DIRNAME,
                            str(self.measurements),
                            self.name,
                            filename)
        return file

    def _value_from_file_cache(self, filename, calculate_method, derivative_used=True, save_as_txt=True, save_as_np=False):
        filename = self._filename(filename)
        file = self.model._cache.get_file(filename, derivative_used=derivative_used)
        value = self.model._cache.get_value(file, calculate_method, save_as_txt=save_as_txt, save_as_np=save_as_np)
        return value

    def _value_in_file_cache(self, filename, derivative_used=True):
        filename = self._filename(filename)
        file = self.model._cache.get_file(filename, derivative_used=derivative_used)
        value = self.model._cache.has_value(file)
        return value

    # model and data values

    @util.cache.memory.method_decorator()
    def model_f(self):
        f = self.model.f_measurements(*self.measurements)
        f = self.measurements.convert_measurements_dict_to_array(f)
        assert len(f) == self.measurements.number_of_measurements
        return f

    @util.cache.memory.method_decorator()
    def model_f_all_boxes(self, time_dim, as_shared_array=False):
        f = self.model.f_all(time_dim, return_as_dict=False)
        assert f.shape[1] == time_dim
        if as_shared_array:
            f = util.parallel.with_multiprocessing.shared_array(f)
        return f

    @util.cache.memory.method_decorator(maxsize=2)
    def model_df(self, derivative_order=1):
        df = self.model.df_measurements(*self.measurements, include_total_concentration=self.include_initial_concentrations_factor_to_model_parameters, derivative_order=derivative_order)
        df = self.measurements.convert_measurements_dict_to_array(df)
        assert df.shape == (self.measurements.number_of_measurements,) + (self.model_parameters_len,)*derivative_order
        return df

    @util.cache.memory.method_decorator(maxsize=2)
    def model_df_all_boxes(self, time_dim, derivative_order=1, as_shared_array=False):
        df = self.model.df_all(time_dim, include_total_concentration=self.include_initial_concentrations_factor_to_model_parameters, derivative_order=derivative_order, return_as_dict=False)
        assert df.shape[1] == time_dim and df.shape[-1] == self.model_parameters_len
        if as_shared_array:
            df = util.parallel.with_multiprocessing.shared_array(df)
        return df

    @util.cache.memory.method_decorator()
    def measurements_results(self):
        measurements_results = self.measurements.values
        assert len(measurements_results) == self.measurements.number_of_measurements
        return measurements_results
