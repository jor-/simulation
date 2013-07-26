import numpy as np
# from pylab import *
import matplotlib.pyplot as pp
# import matplotlib.colors
# import math
import os.path

import multiprocessing

from time import strptime

import ndop.metos3d.data
import ndop.metos3d.direct_access
from ndop.analysis.accuracy_cached import Accuracy_Cached

import util.pattern
from util.debug import Debug



class Logic(Debug):
    
    def __init__(self, debug_level=0):
#         from ndop.plot.constants import LSM_FILE, NOBS_FILE, VARIS_FILE
        
        Debug.__init__(self, debug_level, base_string='ndop.plot.logic: ')
        
        self._df = None
        
        self._parameter_set = 0
        self.set_parameter_set(0)
        self._tracer_index = 0
        self._time_index = -1
        self._depth_index = 0
        self._plot_index = 0
        
        # plot NANs in black
        colormap = pp.cm.jet
        colormap.set_bad('k',1.)
        
#         # plot lsm
# #         lsm = np.load(LSM_FILE)
#         lsm = ndop.metos3d.data.load_land_sea_mask(self.debug_level, self.required_debug_level)
#         lsm = lsm.view(dtype=np.float)
#         lsm[np.where(lsm == 0)] = np.nan
# #         lsm = lsm * 0
#         pp.imshow(lsm.transpose(), origin='lower', extent=(0, 360, -90, 90), vmin=0, vmax=15)
#         self._colorbar = pp.colorbar()
#         
#         pp.show(block=False)
        
        self._accuracy_object = Accuracy_Cached(self.debug_level, self.required_debug_level + 1)
    
# #     @property
#     def get_parameter_set(self):
#         return self._parameter_set
        
    def get_parameter_set_dir(self):
        return self._parameter_set_dir
    
#     @parameter_set.setter
    def set_parameter_set(self, parameter_set):
        from ndop.metos3d.constants import MODEL_OUTPUTS_DIR, MODEL_PARAMETERS_SET_DIRNAME
        
        parameter_set_dir_pattern = os.path.join(MODEL_OUTPUTS_DIR, MODEL_PARAMETERS_SET_DIRNAME)
        parameter_set_dir = util.pattern.replace_int_pattern(parameter_set_dir_pattern, parameter_set)
        self._parameter_set_dir = parameter_set_dir
#         if parameter_set != self._parameter_set:
#             self._parameter_set = parameter_set
#             self._df = None
        
        self.print_debug_inc_dec(('Parameter set dir changed to "', parameter_set_dir, '".'))
        
#         self.parameter_str_changed(str(parameter_set))
    
    
#     @property
    def get_tracer_index(self):
        return self._tracer_index
    
#     @tracer_index.setter
    def set_tracer_index(self, tracer_index):
        self._tracer_index = tracer_index
        
        self.print_debug_inc_dec(('Tracer index changed to "', tracer_index, '".'))
    
    
#     @property
    def get_time_index(self):
        return self._time_index
    
#     @time_index.setter
    def set_time_index(self, time_index):
        time_index = time_index - 1
        self._time_index = time_index
            
        self.print_debug_inc_dec(('Time index changed to "', time_index, '".'))
    
    
#     @property
    def get_depth_index(self):
        return self._depth_index
    
#     @depth_index.setter
    def set_depth_index(self, depth_index):
        self._depth_index = depth_index
        
        self.print_debug_inc_dec(('Depth index changed to "', depth_index, '".'))
    
    
#     @property
    def get_plot_index(self):
        return self._plot_index
    
#     @plot_index.setter
    def set_plot_index(self, plot_index):
#         if plot_index == 0:
#             plot_index = range(self.sensitivity_length)
#         else:
#         plot_index = plot_index - 1
        self._plot_index = plot_index
        
        self.print_debug_inc_dec(('Sensitivity index changed to "', plot_index, '".'))
    
    
#     @staticmethod
#     def parameter_str_changed(self, str):
#         print('BABABBABB')
    
    
    
    
#     def set_time(self, time):
#         time = str(time)
#         self.print_debug_inc_dec(('Time changed to "', time, '".'))
#         
#         if time == "annual":
#             time_index = range(13)
#         else:
#             time_index = strptime(time,'%B').tm_mon
#             
#         self.print_debug_inc_dec(('tracer index changed to: ', time_index))
#         self._tracer_index = time_index

#     def get_parameters_str(self):
#         from ndop.plot.constants import PARAMETER_FILE_PATTERN
#         
#         parameter_set = self.get_parameter_set()
#         parameter_file = util.pattern.replace_int_pattern(PARAMETER_FILE_PATTERN, parameter_set)
#         parameters = np.loadtxt(parameter_file)
#         
#         parameters_strings = ["%.3f" % parameter for parameter in parameters]
#         parameters_string = ''
#         
#         last_index = len(parameters_strings) - 1
#         for i in range(last_index):
#             parameters_string += parameters_strings[i] + '\n'
#         parameters_string += parameters_strings[last_index]
#         
#         return parameters_string

    def get_parameters_strings(self):
        from ndop.metos3d.constants import MODEL_PARAMETERS_FILENAME
        
        parameter_set_dir = self.get_parameter_set_dir()
        parameter_file = os.path.join(parameter_set_dir, MODEL_PARAMETERS_FILENAME)
        parameters = np.loadtxt(parameter_file)
        
        parameters_strings = ["%.3f" % parameter for parameter in parameters]
        
        return parameters_strings
        
    
    def draw_plot(self):
        self.print_debug_inc('Drawing plot.')
        
        parameter_set_dir = self.get_parameter_set_dir()
        tracer_index = self.get_tracer_index()
        depth_index = self.get_depth_index()
        time_index = self.get_time_index()
        plot_index = self.get_plot_index()
        
        
        self.print_debug(('Drawing plot with index "', plot_index, '".'))
        
        ## chose plot map
        MODEL_OUTPUT_PLOT_INDEX = 0
        MODEL_ACCURACY_PLOT_INDEX = 1
        MODEL_DIFF_PLOT = 2
        MEANS_PLOT_INDEX = 3
        NOBS_PLOT_INDEX = 4
        VARIS_OF_OBSERVATION_PLOT_INDEX = 5
        VARIS_OF_OBSERVATION_MEAN_PLOT_INDEX = 6
        PROPABILITY_PLOT_INDEX = 7
        SENSITIVITY_PLOT_INDEX = 8
        
        if plot_index == MODEL_OUTPUT_PLOT_INDEX:
            self.print_debug('Drawing model output plot.')
            map = ndop.metos3d.direct_access.get_f(parameter_set_dir)
        elif plot_index == MODEL_ACCURACY_PLOT_INDEX:
            self.print_debug('Drawing model accuracy plot.')
            map = self._accuracy_object.confidence_for_model(parameter_set_dir)
        elif plot_index == MODEL_DIFF_PLOT:
            self.print_debug('Drawing model diff plot.')
            f = ndop.metos3d.direct_access.get_f(parameter_set_dir)
            y = self._accuracy_object.means
            map = abs(y - f)
        elif plot_index == MEANS_PLOT_INDEX:
            self.print_debug('Drawing observation means plot.')
            map = self._accuracy_object.means
        elif plot_index == NOBS_PLOT_INDEX:
            self.print_debug('Drawing nobs plot.')
            map = self._accuracy_object.nobs
        elif plot_index == VARIS_OF_OBSERVATION_PLOT_INDEX:
            self.print_debug('Drawing varis of observations plot.')
            map = self._accuracy_object.varis
        elif plot_index == VARIS_OF_OBSERVATION_MEAN_PLOT_INDEX:
            self.print_debug('Drawing varis of mean of observations plot.')
            map = self._accuracy_object.vari_of_means
        elif plot_index == PROPABILITY_PLOT_INDEX:
            self.print_debug('Drawing probility plot.')
            map = self._accuracy_object.probability_of_observations(parameter_set_dir)
        else:
            self.print_debug('Drawing sensitivity plot.')
            map = ndop.metos3d.direct_access.get_df(parameter_set_dir)
            map = np.abs(map)
            
            ## chose parameter for sensitivity plot
            if plot_index >= SENSITIVITY_PLOT_INDEX:
                self.print_debug(('Choosing sensitivity index "', plot_index - SENSITIVITY_PLOT_INDEX, '".'))
                if plot_index == SENSITIVITY_PLOT_INDEX:
                    map = np.sum(map, -1)
                elif plot_index > SENSITIVITY_PLOT_INDEX:
                    map = map[..., plot_index - (SENSITIVITY_PLOT_INDEX + 1)]
        
        self.print_debug(('Map shape is "', map.shape, '".'))
        
        
        ## chose tracer, depth and time
        self.print_debug(('Choosing tracer index "', tracer_index, '" depth index "', depth_index, '".'))
        
        map = map[tracer_index, :, :, :, depth_index]
        
        self.print_debug(('Map shape is "', map.shape, '".'))
        
        if time_index >= 0:
            self.print_debug(('Choosing time index "', time_index, '".'))
            map = map[time_index]
        else:
            t_dim = map.shape[0]
            self.print_debug(('Averaging time.'))
            numbers = np.sum(np.logical_not(np.isnan(map)), 0)
            map = np.nansum(map, 0) / numbers
        
        self.print_debug(('Map shape is "', map.shape, '".'))
        
#         vmin = 0    #math.floor(np.nanmin(map))
#         vmax = math.ceil(np.nanmax(map))
        vmin = np.nanmin(map)
        vmax = np.nanmax(map)
        vmax = 1
        if vmax == vmin:
            vmax = vmin + 1
        
#         axes_image = pp.imshow(map.transpose(), origin='lower', extent=(0, 360, -90, 90), vmin=vmin, vmax=vmax, aspect='equal')
        
        try:
            self._colorbar
            is_initiated = True
        except AttributeError:
            is_initiated = False
        
        if is_initiated:
            self.print_debug('Drawing plot. (Plot was initialised before.)')
            axes_image = pp.imshow(map.transpose(), origin='lower', extent=(0, 360, -90, 90), vmin=vmin, vmax=vmax, aspect='equal')
#             axes_image = pp.imshow(map.transpose(), origin='lower', extent=(0, 360, -90, 90), norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax), aspect='equal')
            
            self._colorbar.update_bruteforce(axes_image)
#             pp.draw()
        else:
            self.print_debug('Initialising and drawing plot.')
            
            figure = pp.figure(figsize=(14, 6))
            figure.show()
            
#             axes_image = pp.matshow(map.transpose(), origin='lower', extent=(0, 360, -90, 90), vmin=vmin, vmax=vmax, aspect='equal')
#             self._colorbar = pp.colorbar(axes_image)
# #             pp.show(block=False)
#             pp.show()
            axes_image = pp.imshow(map.transpose(), origin='lower', extent=(0, 360, -90, 90), vmin=vmin, vmax=vmax, aspect='equal')
#             axes_image = pp.imshow(map.transpose(), origin='lower', extent=(0, 360, -90, 90), norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax), aspect='equal')
            self._colorbar = pp.colorbar(axes_image)
            pp.tight_layout()
#             pp.show(block=False)
#             pp.show()
            
#             p = multiprocessing.Process(target=plot_init)
#             p.start()
            
        pp.draw()
            
        
        self.print_debug_dec('Plot drawn.')
#         ion()
#         matshow(sensitivity_map.transpose(), origin='lower', extent=(0, 360, -90, 90))
#         colorbar()
#         show(block=False)
#         draw()
#         show()
#         show(block=False)
#         J
#         J_tracer = J[tracer_index]
#         
#         time_index = self.get_time_index()
#         if time_index >= 0:
#             J_tracer_time = J_tracer[time_index]
#         else:
#             J_tracer_time = np.sum(J_tracer, 0)
#         
#         sensitivity_tracer = np.sum(np.abs(J_tracer), 4)
#         sensitivity_tracer_annual = np.sum(sensitivity_tracer, 0)
#         
#         imshow(sensitivity_tracer_annual[0], origin='lower', extent=(0, 360, -90, 90))
#         title('DOP')
#         colorbar()
#         show()
    