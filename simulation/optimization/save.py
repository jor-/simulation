# *** main function for script call *** #


def _main():

    import argparse
    import os

    import numpy as np

    import simulation
    import simulation.model.constants
    import simulation.optimization.constants
    import simulation.optimization.cost_function
    import simulation.optimization.job
    import simulation.util.args

    import util.logging

    from simulation.optimization.constants import COST_FUNCTION_NAMES

    # parse arguments
    parser = argparse.ArgumentParser(description='Evaluating a cost function for matlab.')

    simulation.util.args.argparse_add_model_options(parser)
    simulation.util.args.argparse_add_measurement_options(parser)

    parser.add_argument('--cost_function_name', required=True, choices=COST_FUNCTION_NAMES, help='The cost function which should be evaluated.')

    parser.add_argument('--eval_f', action='store_true', help='Save the value of the cost function.')
    parser.add_argument('--eval_df', action='store_true', help='Save the values of the derivative of the cost function.')
    parser.add_argument('--eval_d2f', action='store_true', help='Save the values of the second derivative of the cost function.')
    parser.add_argument('--as_job', action='store_true', help='Eval as batch jobn.')

    parser.add_argument('--nodes_setup_node_kind', default=None, help='The node kind to use for the spinup.')
    parser.add_argument('--nodes_setup_number_of_nodes', type=int, default=0, help='The number of nodes to use for the spinup.')
    parser.add_argument('--nodes_setup_number_of_cpus', type=int, default=0, help='The number of cpus to use for the spinup.')

    parser.add_argument('--version', action='version', version='%(prog)s {}'.format(simulation.__version__))

    args = parser.parse_args()

    model_options = simulation.util.args.parse_model_options(args, concentrations_must_be_set=True, parameters_must_be_set=True)
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

    # choose cost function
    cost_function_name = args.cost_function_name
    try:
        cf_class = getattr(simulation.optimization.cost_function, cost_function_name)
    except AttributeError:
        raise ValueError('Unknown cost function {}.'.format(cost_function_name))

    # run cost function evaluation
    with util.logging.Logger():

        # init cost function
        cf = cf_class(measurements_object=measurements_object, model_options=model_options, model_job_options=prepare_model_job_options())

        # if necessary start calculation job
        eval_f = args.eval_f and not cf.f_available()
        eval_df = args.eval_df and not cf.df_available(derivative_order=1)
        eval_d2f = args.eval_d2f and not cf.df_available(derivative_order=2)
        if eval_f or eval_df or eval_d2f:

            # start spinup job
            cf.model.run_dir

            # start cf calculation job
            if args.as_job:
                with simulation.optimization.job.CostFunctionJob(
                        cost_function_name, model_options,
                        model_job_options=prepare_model_job_options(),
                        min_measurements_standard_deviations=args.min_measurements_standard_deviations,
                        min_measurements_correlations=args.min_measurements_correlations,
                        min_standard_deviations=args.min_standard_deviations,
                        correlation_decomposition_min_value_D=args.correlation_decomposition_min_value_D,
                        correlation_decomposition_min_abs_value_L=args.correlation_decomposition_min_abs_value_L,
                        max_box_distance_to_water=args.max_box_distance_to_water,
                        eval_f=eval_f,
                        eval_df=eval_df,
                        eval_d2f=eval_d2f,
                        include_initial_concentrations_factor_to_model_parameters=cf.include_initial_concentrations_factor_to_model_parameters,
                        remove_output_dir_on_close=True) as cf_job:
                    cf_job.start()
            else:
                if eval_f:
                    cf.f()
                if eval_df:
                    cf.df(derivative_order=1)
                if eval_d2f:
                    cf.df(derivative_order=2)


if __name__ == "__main__":
    _main()
