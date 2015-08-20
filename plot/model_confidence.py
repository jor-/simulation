import argparse

from util.logging import Logger
from ndop.plot.interface import model_confidence


if __name__ == "__main__":
    with Logger():
        parser = argparse.ArgumentParser(description='Plotting model output.')
        parser.add_argument('parameter_set', type=int, default=184)
        parser.add_argument('kind', default='WOD_WLS')
        parser.add_argument('--version', action='version', version='%(prog)s 0.1')
        args = vars(parser.parse_args())
        parameter_set = args['parameter_set']
        kind = args['kind']

        model_confidence(parameter_set_nr=parameter_set, kind=kind)