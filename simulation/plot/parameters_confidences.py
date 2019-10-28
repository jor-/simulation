def _main():

    import argparse

    import util.logging
    import simulation.util.args
    import simulation.plot.model

    # init arguments
    parser = argparse.ArgumentParser(description='Plotting parameters confidences.')
    parser = simulation.util.args.argparse_add_accuracy_object_options(parser)
    parser.add_argument('--matrix_type', default='F_H', choices=('F_H', 'F', 'H'), help='The covariance approximation.')
    parser.add_argument('--alpha', type=float, default=0.99, help='The confidence niveau.')
    parser.add_argument('--absolute', action='store_true', help='Use absolute confidence.')
    parser.add_argument('--not_include_variance_factor', action='store_true', help='Do not include variance factor.')
    parser.add_argument('--not_use_interval_length', action='store_true', help='Do not use interval length.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files.')
    parser.add_argument('--kwargs', nargs=argparse.REMAINDER, help='Additional keyword arguments for plots.')
    parser.add_argument('--debug', action='store_true', help='Print debug infos.')

    # parse arguments
    args = parser.parse_args()
    accuracy_object = simulation.util.args.parse_accuracy_object_options(args, concentrations_must_be_set=True, parameters_must_be_set=True)
    if args.kwargs is not None:
        kwargs = dict(kwarg.split('=') for kwarg in args.kwargs)
    else:
        kwargs = {}

    # plot
    with util.logging.Logger(disp_stdout=args.debug):
        simulation.plot.model.parameters_confidences(
            accuracy_object, matrix_type=args.matrix_type, alpha=args.alpha,
            relative=not args.absolute, include_variance_factor=not args.not_include_variance_factor, use_interval_length=not args.not_use_interval_length,
            overwrite=args.overwrite, **kwargs)


if __name__ == "__main__":
    _main()
