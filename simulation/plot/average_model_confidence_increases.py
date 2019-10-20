def _main():

    import argparse

    import util.logging
    import simulation.util.args
    import simulation.plot.model

    # init arguments
    parser = argparse.ArgumentParser(description='Plotting average model confidence increases.')
    parser = simulation.util.args.argparse_add_accuracy_object_options(parser)

    parser.add_argument('--number_of_measurements', type=int, default=1, help='The number of additional measurements.')
    parser.add_argument('--alpha', type=float, default=0.99, help='The confidence niveau.')
    parser.add_argument('--not_include_variance_factor', action='store_true', help='Do not include varaiance factor.')
    parser.add_argument('--not_relative', action='store_true', help='Do not use relative average model confidence but absolute average model confidence.')
    parser.add_argument('--time_dim_model', type=int, default=12, help='The time dimension used for the model.')
    parser.add_argument('--time_dim_confidence_increase', type=int, default=12, help='The time dimension used for the model confidence increase.')
    parser.add_argument('--tracer', default=None, help='The tracer that should be ploted. If not passed, all tracers are plotted.')

    parser.add_argument('--plot_type', default='all', help='Desired plot type.')
    parser.add_argument('--v_max', default=None, help='The maximal value used in the plot.')
    parser.add_argument('--no_colorbar', action='store_true', help='Do not plot colorbar.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files.')
    parser.add_argument('--kwargs', nargs=argparse.REMAINDER, help='Additional keyword arguments for plots.')
    parser.add_argument('--debug', action='store_true', help='Print debug infos.')

    # parse arguments
    args = parser.parse_args()
    accuracy_object = simulation.util.args.parse_accuracy_object_options(args, concentrations_must_be_set=True, parameters_must_be_set=True)

    v_max = args.v_max
    if v_max is not None:
        try:
            v_max = float(v_max)
        except ValueError:
            pass

    if args.kwargs is not None:
        kwargs = dict(kwarg.split('=') for kwarg in args.kwargs)
    else:
        kwargs = {}

    # plot
    with util.logging.Logger(disp_stdout=args.debug):
        simulation.plot.model.confidence_increases(
            accuracy_object, number_of_measurements=args.number_of_measurements, alpha=args.alpha,
            include_variance_factor=not args.not_include_variance_factor, relative=not args.not_relative,
            time_dim_model=args.time_dim_model, time_dim_confidence_increase=args.time_dim_confidence_increase,
            tracer=args.tracer, plot_type=args.plot_type, v_max=v_max, colorbar=not args.no_colorbar, overwrite=args.overwrite, **kwargs)


if __name__ == "__main__":
    _main()
