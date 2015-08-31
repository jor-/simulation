if __name__ == "__main__":
    import argparse
    
    from util.logging import Logger
    from ndop.plot.interface import model_output

    with Logger():
        parser = argparse.ArgumentParser(description='Plotting model output.')
        parser.add_argument('parameter_set', type=int, default=0)
        parser.add_argument('kind', choices=['BOXES', 'WOD'], default='BOXES')
        parser.add_argument('-v', '--v_max', type=float, nargs=2)
        parser.add_argument('--version', action='version', version='%(prog)s 0.1')

        args = vars(parser.parse_args())
        parameter_set = args['parameter_set']
        kind = args['kind']
        v_max = args['v_max']
        if v_max is None:
            v_max = (None, None)

        model_output(parameter_set_nr=parameter_set, kind=kind, y_max=v_max)