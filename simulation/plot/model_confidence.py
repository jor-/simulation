import argparse

from util.logging import Logger
from simulation.plot.interface import model_confidence


if __name__ == "__main__":
    with Logger():
        parser = argparse.ArgumentParser(description='Plotting model output.')
        parser.add_argument('parameter_set', type=int, default=184)
        parser.add_argument('kind', default='WOD_WLS')
        parser.add_argument('--average_in_time', '-a', action='store_true')
        parser.add_argument('--max_dop', type=float, default=None)
        parser.add_argument('--max_po4', type=float, default=None)
        parser.add_argument('--version', action='version', version='%(prog)s 0.1')
        args = parser.parse_args()
        
        model_confidence(parameter_set_nr=args.parameter_set, kind=args.kind, average_in_time=args.average_in_time, v_max=[args.max_dop, args.max_po4])