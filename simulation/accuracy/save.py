import multiprocessing  # for using multiprocessing logger

import util.logging

import measurements.all.data

import simulation.accuracy.linearized
import simulation.util.args


def save(model_options, measurements_object, cost_function_name,
         alpha=0.99, time_dim_model=None, parallel=True):

    # create accuracy object
    if cost_function_name == 'OLS':
        accuracy_class = simulation.accuracy.linearized.OLS
    elif cost_function_name == 'WLS':
        accuracy_class = simulation.accuracy.linearized.WLS
    elif cost_function_name == 'GLS':
        accuracy_class = simulation.accuracy.linearized.GLS
    else:
        raise ValueError(f'Unknown cost_function_name {cost_function_name}.')
    accuracy_object = accuracy_class(measurements_object, model_options=model_options)

    for matrix_type in ('F', 'H', 'F_H'):
        accuracy_object.correlation_matrix(matrix_type=matrix_type)
        for include_variance_factor in (True, False):
            accuracy_object.covariance_matrix(matrix_type=matrix_type, include_variance_factor=include_variance_factor)
            for time_dim_confidence in (12, 4, 1):
                accuracy_object.model_confidence(matrix_type=matrix_type, include_variance_factor=include_variance_factor, alpha=alpha, time_dim_confidence=time_dim_confidence, time_dim_model=time_dim_model, parallel=parallel)
            for relative in (True, False):
                accuracy_object.parameter_confidence(matrix_type=matrix_type, include_variance_factor=include_variance_factor, alpha=alpha, relative=relative)
                for per_tracer in (True, False):
                    accuracy_object.average_model_confidence(matrix_type=matrix_type, include_variance_factor=include_variance_factor, alpha=alpha, time_dim_model=time_dim_model, per_tracer=per_tracer, relative=relative, parallel=parallel)
    for time_dim_confidence in (4, 1, 12):
        accuracy_object.confidence_increase(alpha=alpha, time_dim_confidence_increase=time_dim_confidence, time_dim_model=time_dim_model, increases_calculation_relative=True, include_variance_factor=True, parallel=parallel, number_of_measurements=1)


# *** main function for script call *** #

def _main():

    # parse arguments
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate and save model accuracy.')

    parser.add_argument('cost_function_name', choices=('OLS', 'WLS', 'GLS'), default=None, help='The cost functions to evaluate.')

    simulation.util.args.argparse_add_model_options(parser)
    simulation.util.args.argparse_add_measurement_options(parser)

    parser.add_argument('--alpha', type=float, default=0.99, help='The confidence level.')
    parser.add_argument('--time_dim_model', type=int, default=None, help='Used time dim of model.')
    parser.add_argument('--parallel', action='store_true', help='Run in parallel.')

    parser.add_argument('-d', '--debug', action='store_true', help='Print debug infos.')

    parser.add_argument('--version', action='version', version='%(prog)s {}'.format(simulation.__version__))

    args = parser.parse_args()

    # call function
    with util.logging.Logger(disp_stdout=args.debug):
        model_options = simulation.util.args.parse_model_options(args, concentrations_must_be_set=True, parameters_must_be_set=True)
        measurements_object = simulation.util.args.parse_measurements_options(args, model_options)
        save(model_options, measurements_object, args.cost_function_name,
             alpha=args.alpha, time_dim_model=args.time_dim_model, parallel=args.parallel)
        util.logging.info('Finished.')


if __name__ == "__main__":
    _main()
