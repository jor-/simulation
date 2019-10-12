def _main():

    import argparse
    import os

    import numpy as np

    import simulation
    import simulation.accuracy.linearized
    import simulation.util.args
    import simulation.plot.model

    import util.logging

    from simulation.optimization.constants import COST_FUNCTION_NAMES

    # init arguments
    parser = argparse.ArgumentParser(description='Plotting parameters confidences.')

    simulation.util.args.argparse_add_model_options(parser)
    simulation.util.args.argparse_add_measurement_options(parser)

    parser.add_argument('--cost_function_name', required=True, choices=COST_FUNCTION_NAMES, help='The cost function which should be evaluated.')
    parser.add_argument('--alpha', type=float, default=0.99, help='The confidence niveau.')
    parser.add_argument('--matrix_type', default='F_H', choices=('F_H', 'F', 'H'), help='The covariance approximation.')
    parser.add_argument('--absolute', action='store_true', help='Use absolute confidence.')
    parser.add_argument('--not_include_variance_factor', action='store_true', help='Do not include varaiance factor.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files.')
    parser.add_argument('--kwargs', nargs=argparse.REMAINDER, help='Additional keyword arguments for plots.')

    # parse arguments
    args = parser.parse_args()
    model_options = simulation.util.args.parse_model_options(args, concentrations_must_be_set=True, parameters_must_be_set=True)
    measurements_object = simulation.util.args.parse_measurements_options(args, model_options)

    if args.kwargs is not None:
        kwargs = dict(kwarg.split('=') for kwarg in args.kwargs)
    else:
        kwargs = {}

    # init accuracy class
    cost_function_name = args.cost_function_name
    try:
        accuracy_class = getattr(simulation.accuracy.linearized, cost_function_name)
    except AttributeError:
        raise ValueError('Unknown accuracy class {}.'.format(cost_function_name))
    accuracy_object = accuracy_class(measurements_object=measurements_object, model_options=model_options)

    # plot
    simulation.plot.model.parameters_confidences(accuracy_object, alpha=args.alpha, relative=not args.absolute, matrix_type=args.matrix_type, include_variance_factor=not args.not_include_variance_factor, overwrite=args.overwrite, **kwargs)


if __name__ == "__main__":
    _main()
