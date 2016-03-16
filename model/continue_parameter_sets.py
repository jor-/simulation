import sys
import argparse
import signal
import os
import logging
import numpy as np

from ndop.model.eval import Model

# import util.pattern
import util.io.fs


class Continue():

    def __init__(self):
        self.continue_execution = True
        signal.signal(signal.SIGTERM, self.stop)

    def stop(self):
        self.continue_execution = False



    def continue_parameter_sets(self, years, tolerance, time_step, parameter_set_numbers=None):
        from ndop.model.constants import MODEL_OUTPUT_DIR, DATABASE_TIME_STEP_DIRNAME, DATABASE_PARAMETERS_SET_DIRNAME, DATABASE_PARAMETERS_FILENAME

        logging.debug('Continue runs with years={} tolerance={} and time_step={}.'.format(years, tolerance, time_step))

        model = Model()

#         time_step_dirname = util.pattern.replace_int_pattern(DATABASE_TIME_STEP_DIRNAME, time_step)
        time_step_dirname = DATABASE_TIME_STEP_DIRNAME.format(time_step)
        time_step_dir = os.path.join(MODEL_OUTPUT_DIR, time_step_dirname)

        if parameter_set_numbers is None:
            parameter_set_dirs = util.io.fs.get_dirs(time_step_dir)
            parameter_set_numbers = range(len(parameter_set_dirs))

        logging.debug('Continue runs for parameter set with number {}.'.format(parameter_set_numbers))

        for parameter_set_number in parameter_set_numbers:
            if self.continue_execution:
                logging.debug('Continue runs for parameter set with number {}.'.format(parameter_set_numbers))

#                 parameter_set_dirname = util.pattern.replace_int_pattern(DATABASE_PARAMETERS_SET_DIRNAME, parameter_set_number)
                parameter_set_dirname = DATABASE_PARAMETERS_SET_DIRNAME.format(parameter_set_number)
                parameter_set_dir = os.path.join(time_step_dir, parameter_set_dirname)

                parameters_file = os.path.join(parameter_set_dir, DATABASE_PARAMETERS_FILENAME)
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
    parser.add_argument('--version', action='version', version='%(prog)s 0.1')
    args = vars(parser.parse_args())

    parameter_set_numbers = range(args['min'], args['max'] + 1)

    continue_object = Continue()

    continue_object.continue_parameter_sets(years=args['years'], tolerance=args['tolerance'], time_step=args['time_step'], parameter_set_numbers=parameter_set_numbers)

    sys.exit(0)