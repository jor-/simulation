# *** main function for script call *** #


def _main():

    import argparse
    import os

    import numpy as np

    import simulation
    import simulation.model.constants
    import simulation.model.options
    import simulation.optimization.cost_function
    import simulation.optimization.job
    import simulation.util.args

    import measurements.all.data

    import util.batch.universal.system
    import util.io.matlab
    import util.io.fs

    import util.logging

    from simulation.optimization.matlab.constants import MATLAB_PARAMETER_FILENAME, MATLAB_F_FILENAME, MATLAB_DF_FILENAME, NODES_MAX_FILENAME, COST_FUNCTION_NAMES

    # parse arguments
    parser = argparse.ArgumentParser(description='Evaluating a cost function for matlab.')

    simulation.util.args.argparse_add_model_options(parser)
    simulation.util.args.argparse_add_measurement_options(parser)

    parser.add_argument('--cost_function_name', required=True, choices=COST_FUNCTION_NAMES, help='The cost function which should be evaluated.')

    parser.add_argument('--exchange_dir', required=True, help='The directory from where to load the parameters and where to save the cost function values.')
    parser.add_argument('--debug_logging_file', default=None, help='File to store debug informations.')

    parser.add_argument('--eval_function_value', action='store_true', help='Save the value of the cost function.')
    parser.add_argument('--eval_grad_value', action='store_true', help='Save the values of the derivative of the cost function.')

    parser.add_argument('--nodes_setup_node_kind', default=None, help='The node kind to use for the spinup.')
    parser.add_argument('--nodes_setup_number_of_nodes', type=int, default=0, help='The number of nodes to use for the spinup.')
    parser.add_argument('--nodes_setup_number_of_cpus', type=int, default=0, help='The number of cpus to use for the spinup.')

    parser.add_argument('--version', action='version', version='%(prog)s {}'.format(simulation.__version__))

    args = parser.parse_args()

    model_options = simulation.util.args.parse_model_options(args, concentrations_must_be_set=True, parameters_must_be_set=False)
    measurements_object = simulation.util.args.parse_measurements_options(args, model_options)

    # set job setup
    def prepare_model_job_options():
        if args.nodes_setup_node_kind is not None:
            nodes_setup = simulation.model.constants.NODES_SETUP_SPINUP.copy()
            nodes_setup['node_kind'] = args.nodes_setup_node_kind
            nodes_setup['nodes'] = args.nodes_setup_number_of_nodes
            nodes_setup['cpus'] = args.nodes_setup_number_of_cpus
            job_options = {'spinup': {'nodes_setup': nodes_setup}}
        else:
            job_options = None
        return job_options

    # calculate file locations
    exchange_dir = args.exchange_dir
    p_file = os.path.join(exchange_dir, MATLAB_PARAMETER_FILENAME)
    f_file = os.path.join(exchange_dir, MATLAB_F_FILENAME)
    df_file = os.path.join(exchange_dir, MATLAB_DF_FILENAME)

    # load cf parameters
    parameters = util.io.matlab.load(p_file, 'p')

    # choose cost function
    cost_function_name = args.cost_function_name
    try:
        cf_class = getattr(simulation.optimization.cost_function, cost_function_name)
    except AttributeError:
        raise ValueError('Unknown cost function {}.'.format(cost_function_name))

    # run cost function evaluation
    log_file = args.debug_logging_file
    with util.logging.Logger(log_file=log_file, disp_stdout=log_file is None):

        # init cost function
        cf = cf_class(measurements_object=measurements_object, model_options=model_options, model_job_options=prepare_model_job_options())
        cf.model_parameters = parameters

        # if necessary start calculation job
        eval_function_value = args.eval_function_value
        eval_grad_value = args.eval_grad_value
        if (eval_function_value and not cf.f_available()) or (eval_grad_value and not cf.df_available()):

            # start spinup job
            cf.model.run_dir

            # start cf calculation job
            with simulation.optimization.job.CostFunctionJob(
                    cost_function_name, model_options,
                    model_job_options=prepare_model_job_options(),
                    min_measurements_standard_deviations=args.min_measurements_standard_deviations,
                    min_measurements_correlations=args.min_measurements_correlations,
                    min_standard_deviations=args.min_standard_deviations,
                    correlation_decomposition_min_value_D=args.correlation_decomposition_min_value_D,
                    correlation_decomposition_min_abs_value_L=args.correlation_decomposition_min_abs_value_L,
                    max_box_distance_to_water=args.max_box_distance_to_water,
                    eval_f=eval_function_value,
                    eval_df=eval_grad_value,
                    include_initial_concentrations_factor_to_model_parameters=cf.include_initial_concentrations_factor_to_model_parameters,
                    remove_output_dir_on_close=True) as cf_job:
                cf_job.start()
                cf_job.wait_until_finished()

        # save cost function values
        if eval_function_value:
            util.io.matlab.save(f_file, cf.f(), value_name='f', oned_as='column')
        if eval_grad_value:
            util.io.matlab.save(df_file, cf.df(), value_name='df', oned_as='column')


if __name__ == "__main__":
    _main()
