if __name__ == "__main__":
    import argparse
    
    from util.logging import Logger
    from simulation.plot.interface import model_output

    with Logger():
        parser = argparse.ArgumentParser(description='Plotting model output.')
        parser.add_argument('parameter_set', type=int, default=0)
        parser.add_argument('kind', choices=['BOXES', 'WOD'], default='BOXES')
        parser.add_argument('--average_in_time', '-a', action='store_true')
        parser.add_argument('-v', '--v_max', type=float, nargs=2)
        
        args = parser.parse_args()
        v_max = args.v_max
        if v_max is None:
            v_max = (None, None)

        model_output(parameter_set_nr=args.parameter_set, kind=args.kind, y_max=v_max, average_in_time=args.average_in_time)