def _main():
    import argparse

    import util.logging
    import simulation.model.constants
    import simulation.plot.model
    import simulation.util.args

    with util.logging.Logger():
        parser = argparse.ArgumentParser(description='Plotting model output.')

        simulation.util.args.argparse_add_model_options(parser)

        parser.add_argument('--tracer', type=str, default=None)
        parser.add_argument('--time_dim', type=int, default=1)
        parser.add_argument('--path', type=str, default=None)
        parser.add_argument('--y_max', type=float, default=None)
        parser.add_argument('--colorbar', action='store_true')
        parser.add_argument('--contours', action='store_true')

        args = parser.parse_args()

        model_options = simulation.util.args.parse_model_options(args)

        simulation.plot.model.output(model_options, tracer=args.tracer, time_dim=args.time_dim, path=args.path, y_max=args.y_max, contours=args.contours, colorbar=args.colorbar)


if __name__ == "__main__":
    _main()
