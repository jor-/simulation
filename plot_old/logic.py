import numpy as np
from pylab import *

from time import strptime

from ndop.util.debug import Debug

class Logic(Debug):

    
    
    def __init__(self, debug_level=0):
        Debug.__init__(self, debug_level, 'ndop.plot.logic: ')
        
        self._J = None
    
        self._parameter_set = 1
        self._tracer_index = 0
        self._time_index = -1
        self._depth_index = 0
        self._sensitivity_index = -1
    
    
    def get_J(self):
        if self._J == None:
            self._J = np.load('/mnt/work_j2_rz/NDOP/oed_data/parameter_set_1/J_12.npy', 'r')
        
        return self._J
    
    def set_parameter_set(self, parameter_set):
        parameter_set = parameter_set + 1
        self._parameter_set = parameter_set
        
        self.print_debug_inc_dec(('Parameter set changed to "', parameter_set, '".'))
    
    def set_tracer_index(self, tracer_index):
        self._tracer_index = tracer_index
        
        self.print_debug_inc_dec(('Tracer index changed to "', tracer_index, '".'))
    
    def get_tracer_index(self):
        return self._tracer_index
    
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


    def set_time_index(self, time_index):
#         if time_index == 0:
#             time_index = range(self.time_length)
#         else:
        time_index = time_index - 1
        self._tracer_index = time_index
            
        self.print_debug_inc_dec(('Time index changed to "', time_index, '".'))
    
    def get_time_index(self):
        return self._time_index
    
    
    def set_depth_index(self, depth_index):
        self._depth_index = depth_index
        
        self.print_debug_inc_dec(('Depth index changed to "', depth_index, '".'))
    
    def get_depth_index(self):
        return self._depth_index
    
    
    def set_sensitivity_index(self, sensitivity_index):
#         if sensitivity_index == 0:
#             sensitivity_index = range(self.sensitivity_length)
#         else:
        sensitivity_index = sensitivity_index - 1
        self._sensitivity_index = sensitivity_index
        
        self.print_debug_inc_dec(('Sensitivity index changed to "', sensitivity_index, '".'))
    
    def get_sensitivity_index(self):
        return self._sensitivity_index
    
    
        
    
    def draw_plot(self):
#         randomNumbers = range(0, 10)
#         self.widget.canvas.ax.clear()
#         self.widget.canvas.ax.plot(randomNumbers)
#         self.widget.canvas.draw()
        J = self.get_J()
        J = np.abs(J)
        
        tracer_index = self.get_tracer_index()
        depth_index = self.get_depth_index()
        time_index = self.get_time_index()
        sensitivity_index = self.get_sensitivity_index()
        
        sensitivity_map = J[tracer_index, :, depth_index, :, :, :]
        
        if time_index >= 0:
            sensitivity_map = sensitivity_map[time_index]
        else:
            sensitivity_map = np.sum(sensitivity_map, 0)
        
        if sensitivity_index >= 0:
            sensitivity_map = sensitivity_map[sensitivity_index]
        else:
            sensitivity_map = np.sum(sensitivity_map, 2)
        
        imshow(sensitivity_map, origin='lower', extent=(0, 360, -90, 90))
        colorbar()
        show()
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
    