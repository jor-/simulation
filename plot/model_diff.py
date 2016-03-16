import argparse

from util.logging import Logger
from simulation.plot.interface import model_diff


if __name__ == "__main__":
    with Logger():
        parser = argparse.ArgumentParser(description='Plotting model measurements diff.')
        parser.add_argument('parameter_set', type=int, default=0)
        parser.add_argument('data_kind', choices=['WOA', 'WOD'], default='WOA')
        parser.add_argument('-n', '--normalize', action='store_true', help='Normalize with deviation.')
        parser.add_argument('-v', '--v_max', type=float, nargs=2)
        parser.add_argument('--version', action='version', version='%(prog)s 0.1')

        args = vars(parser.parse_args())
        parameter_set = args['parameter_set']
        data_kind = args['data_kind']
        normalize = args['normalize']
        v_max = args['v_max']
        if v_max is None:
            v_max = (None, None)

        model_diff(parameter_set_nr=parameter_set, data_kind=data_kind, normalize_with_deviation=normalize, y_max=v_max)

#         model_diff(parameter_set_nr=parameter_set, data_kind=data_kind, normalize_with_deviation=True, y_max=[None, None])