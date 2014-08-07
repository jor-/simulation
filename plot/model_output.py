import argparse

from util.logging import Logger
from ndop.plot.interface import model_output


if __name__ == "__main__":
    with Logger():
        parser = argparse.ArgumentParser(description='Plotting model output.')
        parser.add_argument('parameter_set', type=int, default=0)
        parser.add_argument('--version', action='version', version='%(prog)s 0.1')
        args = vars(parser.parse_args())
        parameter_set = args['parameter_set']
        
        model_output(parameter_set_nr=parameter_set, y_max=[None, 3])