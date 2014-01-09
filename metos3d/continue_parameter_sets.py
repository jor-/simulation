import sys
import argparse
import signal

import os
import numpy as np

from ndop.metos3d.model import Model

import util.pattern
import util.io
from util.debug import print_debug


class Continue():
    
    def __init__(self):
        self.continue_execution = True
        signal.signal(signal.SIGTERM, self.stop)
    
    def stop(self):
        self.continue_execution = False
    


    def continue_parameter_sets(self, years, tolerance, time_step, parameter_set_numbers=None, debug_level=0, required_debug_level=1):
        from ndop.metos3d.constants import MODEL_OUTPUTS_DIR, MODEL_TIME_STEP_DIRNAME, MODEL_PARAMETERS_SET_DIRNAME, MODEL_PARAMETERS_FILENAME
        
        print_debug(('Continue runs with years=', years, ' tolerance=', tolerance, 'and time_step=', time_step, '.'), debug_level, required_debug_level)
        
        model = Model(debug_level, required_debug_level+1)
        
        time_step_dirname = util.pattern.replace_int_pattern(MODEL_TIME_STEP_DIRNAME, time_step)
        time_step_dir = os.path.join(MODEL_OUTPUTS_DIR, time_step_dirname)
        
        if parameter_set_numbers is None:
            parameter_set_dirs = util.io.get_dirs(time_step_dir)
            parameter_set_numbers = range(len(parameter_set_dirs))
        
        print_debug(('Continue runs for parameter set with number ', parameter_set_numbers, '.'), debug_level, required_debug_level)
        
        for parameter_set_number in parameter_set_numbers:
            if self.continue_execution:
                print_debug(('Continue run for parameter set with number ', parameter_set_number, '.'), debug_level, required_debug_level)
                
                parameter_set_dirname = util.pattern.replace_int_pattern(MODEL_PARAMETERS_SET_DIRNAME, parameter_set_number)
                parameter_set_dir = os.path.join(time_step_dir, parameter_set_dirname)
                
                parameters_file = os.path.join(parameter_set_dir, MODEL_PARAMETERS_FILENAME)
                p = np.loadtxt(parameters_file)
                
                model.f(p, years=years, tolerance=tolerance, time_step_size=time_step)




if __name__ == "__main__":
#     signal.signal(signal.SIGTERM, signal_term_handler)
    
    parser = argparse.ArgumentParser(description='Continues runs for parameter sets.')
    parser.add_argument('-y', '--years', default=10000, type=int, help='the years')
    parser.add_argument('-t', '--time_step', default=1, type=int, help='the time step size')
    parser.add_argument('-T', '--tolerance', default=10**(-5), type=float, help='the tolerance')
    parser.add_argument('-m', '--min', default=0, type=int, help='the minimal parameter set number to use')
    parser.add_argument('-M', '--max', default=10000, type=int, help='the minimal parameter set number to use')
    parser.add_argument('-d', '--debug_level', default=0, type=int, help='Increase the debug level for more debug informations.')
    parser.add_argument('--version', action='version', version='%(prog)s 0.1')
    args = vars(parser.parse_args())
    
    parameter_set_numbers = range(args['min'], args['max'] + 1)
    
    continue_object = Continue()
    
    continue_object.continue_parameter_sets(years=args['years'], tolerance=args['tolerance'], time_step=args['time_step'], parameter_set_numbers=parameter_set_numbers, debug_level=args['debug_level'])
    
    sys.exit(0)