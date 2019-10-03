def _main():
    import argparse

    import util.logging
    import simulation.model.constants
    import simulation.plot.model
    import simulation.util.args

    with util.logging.Logger():
        # config args

        parser = argparse.ArgumentParser(description='Plotting model output.')

        simulation.util.args.argparse_add_model_options(parser)

        parser.add_argument('--time_dim', default=12, type=int, help='Desired time dim of model output.')
        parser.add_argument('--plot_type', default='all', help='Desired plot type.')
        parser.add_argument('--v_max', default=None, help='The maximal value used in the plot.')
        parser.add_argument('--no_colorbar', action='store_true', help='Do not plot colorbar.')
        parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files.')
        parser.add_argument('--kwargs', nargs=argparse.REMAINDER, help='Additional keyword arguments for plots.')

        # parse args

        args = parser.parse_args()

        model_options = simulation.util.args.parse_model_options(args, concentrations_must_be_set=True, parameters_must_be_set=True)

        time_dim = args.time_dim
        plot_type = args.plot_type

        v_max = args.v_max
        if v_max is not None:
            try:
                v_max = float(v_max)
            except ValueError:
                pass

        colorbar = not args.no_colorbar

        overwrite = args.overwrite

        if args.kwargs is not None:
            kwargs = dict(kwarg.split('=') for kwarg in args.kwargs)
        else:
            kwargs = {}

        # plot
        simulation.plot.model.output(model_options, time_dim, plot_type=plot_type, v_max=v_max, overwrite=overwrite, colorbar=colorbar, **kwargs)


if __name__ == "__main__":
    _main()
