import argparse

from util.logging import Logger
from ndop.plot.interface import model_diff


if __name__ == "__main__":
    with Logger():
        parser = argparse.ArgumentParser(description='Plotting model measurements diff.')
        parser.add_argument('parameter_set', type=int, default=0)
        parser.add_argument('data_kind', choices=['WOA', 'WOD'], default='WOA')
        parser.add_argument('--version', action='version', version='%(prog)s 0.1')
        args = vars(parser.parse_args())
        parameter_set = args['parameter_set']
        data_kind = args['data_kind']
        
        model_diff(parameter_set_nr=parameter_set, data_kind=data_kind, y_max=[1, 1])