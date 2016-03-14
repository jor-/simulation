if __name__ == "__main__":
    
    import argparse
    import sys
    import os
    import tempfile

    import numpy as np
    
    import ndop.optimization.cost_function
    import ndop.optimization.job
    import util.batch.universal.system
    import util.io.matlab
    import util.io.fs
    
    import util.logging
    logger = util.logging.logger
    
    from ndop.optimization.matlab.constants import MATLAB_PARAMETER_FILENAME, MATLAB_F_FILENAME, MATLAB_DF_FILENAME, NODES_MAX_FILENAME, KIND_OF_COST_FUNCTIONS
    
    
    ## parse arguments
    parser = argparse.ArgumentParser(description='Evaluating a cost function for matlab.')

    parser.add_argument('--kind_of_cost_function', '-c', choices=KIND_OF_COST_FUNCTIONS, help='The cost function which should be evaluated.')
    parser.add_argument('--exchange_dir', '-p', help='The directory from where to load the parameters and where to save the cost function values.')
    parser.add_argument('--debug_logging_file', '-d', default=None, help='File to store debug informations.')

    parser.add_argument('--eval_function_value', '-f', action='store_true', help='Save the value of the cost function.')
    parser.add_argument('--eval_grad_value', '-g', action='store_true', help='Save the values of the derivative of the cost function.')

    parser.add_argument('--spinup_years', '-y', '--years', type=int, default=10000, help='The number of years for the spinup.')
    parser.add_argument('--spinup_tolerance', '-t', '--tolerance', type=float, default=0, help='The tolerance for the spinup.')
    parser.add_argument('--spinup_satisfy_years_and_tolerance', '-a', '--and_combination', action='store_true', help='If used, the spinup is terminated if years and tolerance have been satisfied. Otherwise, the spinup is terminated as soon as years or tolerance have been satisfied.')

    parser.add_argument('--nodes_setup_node_kind', '--node_kind', default=None, help='The node kind to use for the spinup.')
    parser.add_argument('--nodes_setup_number_of_nodes', '--nodes', type=int, default=0, help='The number of nodes to use for the spinup.')
    parser.add_argument('--nodes_setup_number_of_cpus', '--cpus', type=int, default=0, help='The number of cpus to use for the spinup.')

    parser.add_argument('--parameters_relative_tolerance', type=float, nargs='+', default=None, help='The relative tolerance up to which two parameter vectors are considered equal.')
    parser.add_argument('--parameters_absolute_tolerance', type=float, nargs='+', default=None, help='The absolute tolerance up to which two parameter vectors are considered equal.')

    parser.add_argument('--derivative_step_size', type=float, default=None, help='The step size used for the finite difference approximation.')
    parser.add_argument('--derivative_years', type=int, default=None, help='The number of years for the finite difference approximation spinup.')
    parser.add_argument('--derivative_accuracy_order', type=int, default=None, help='The accuracy order used for the finite difference approximation. 1 = forward differences. 2 = central differences.')

    parser.add_argument('--version', action='version', version='%(prog)s 0.1')

    args = parser.parse_args()

    ## check wich values to evaluate
    eval_function_value = args.eval_function_value
    eval_grad_value = args.eval_grad_value

    ## prepare spinup options
    if args.spinup_satisfy_years_and_tolerance:
        combination='and'
    else:
        combination='or'
    spinup_options = {'years': args.spinup_years, 'tolerance': args.spinup_tolerance, 'combination':combination}

    ## prepare derivative options
    derivative_options = {}
    if args.derivative_step_size is not None:
        derivative_options['step_size'] = args.derivative_step_size
    if args.derivative_years is not None:
        derivative_options['years'] = args.derivative_years
    if args.derivative_accuracy_order is not None:
        derivative_options['accuracy_order'] = args.derivative_accuracy_order
    
    ## prepare time step
    time_step = 1
    
    ## prepare parameter tolerance options
    parameter_tolerance_options = {}
    if args.parameters_relative_tolerance is not None:
        assert len(args.parameters_relative_tolerance) in (1, 7)
        parameter_tolerance_options['relative'] = np.array(args.parameters_relative_tolerance)
    if args.parameters_absolute_tolerance is not None:
        assert len(args.parameters_absolute_tolerance) in (1, 7)
        parameter_tolerance_options['absolute'] = np.array(args.parameters_absolute_tolerance)
    
    ## prepare model options
    model_options = {'spinup_options': spinup_options, 'derivative_options': derivative_options, 'time_step': time_step, 'parameter_tolerance_options': parameter_tolerance_options}

    ## prepare job setup
    def prepare_job_setup():
        if args.nodes_setup_node_kind is not None:
            from ndop.optimization.constants import COST_FUNCTION_NODES_SETUP_SPINUP
            nodes_setup = COST_FUNCTION_NODES_SETUP_SPINUP.copy()
            nodes_setup['node_kind'] = args.nodes_setup_node_kind
            nodes_setup['nodes'] = args.nodes_setup_number_of_nodes
            nodes_setup['cpus'] = args.nodes_setup_number_of_cpus
            job_setup = {'spinup':{'nodes_setup':nodes_setup}}
        else:
            job_setup = None
        return job_setup


    ## run cost function evaluation
    log_file = args.debug_logging_file
    with util.logging.Logger(log_file=log_file, disp_stdout=log_file is None):
        with np.errstate(invalid='ignore'):

            ## calculate file locations
            exchange_dir = args.exchange_dir
            p_file = os.path.join(exchange_dir, MATLAB_PARAMETER_FILENAME)
            f_file = os.path.join(exchange_dir, MATLAB_F_FILENAME)
            df_file = os.path.join(exchange_dir, MATLAB_DF_FILENAME)

            ## choose cost function
            kind_of_cost_function = args.kind_of_cost_function
            kind_of_cost_function_splitted = kind_of_cost_function.split('_')
            data_kind = kind_of_cost_function_splitted[0]
            cf_kind_splitted = kind_of_cost_function_splitted[1].split('.')
            cf_kind = cf_kind_splitted[0]

            if cf_kind == 'OLS':
                cf_class = ndop.optimization.cost_function.OLS
            elif cf_kind == 'WLS':
                cf_class = ndop.optimization.cost_function.WLS
            elif cf_kind == 'LWLS':
                cf_class = ndop.optimization.cost_function.LWLS
            elif cf_kind == 'GLS':
                cf_class = ndop.optimization.cost_function.GLS
                correlation_min_values = int(cf_kind_splitted[1])
                correlation_max_year_diff = int(cf_kind_splitted[2])
                if correlation_max_year_diff < 0:
                    correlation_max_year_diff = float('inf')
            else:
                raise ValueError('Unknown cf kind {}.'.format(cf_kind))

            ## init cost function
            cf_kargs = {'data_kind': data_kind, 'model_options': model_options, 'job_setup': prepare_job_setup()}
            if cf_kind == 'GLS':
                cf_kargs['correlation_min_values'] = correlation_min_values
                cf_kargs['correlation_max_year_diff'] = correlation_max_year_diff

            cf = cf_class(**cf_kargs)

            ## load parameter
            parameters = util.io.matlab.load(p_file, 'p')

            ## if necessary start calculation job
            if (eval_function_value and not cf.f_available(parameters)) or (eval_grad_value and not cf.df_available(parameters)):
                from ndop.model.constants import MODEL_START_FROM_CLOSEST_PARAMETER_SET
                from util.constants import TMP_DIR

                ## start spinup job
                cf.data_base.model.spinup_run_dir(parameters, cf.data_base.model.spinup_options, start_from_closest_parameters=MODEL_START_FROM_CLOSEST_PARAMETER_SET)

                ## start cf calculation job
                os.makedirs(TMP_DIR, exist_ok=True)
                output_dir = tempfile.mkdtemp(dir=TMP_DIR, prefix='cost_function_tmp_')
                util.io.fs.add_group_permissions(output_dir)
                cf_kargs['job_setup'] = prepare_job_setup()
                with ndop.optimization.job.CostFunctionJob(output_dir, parameters, cf_kind, eval_f=eval_function_value, eval_df=eval_grad_value, **cf_kargs) as cf_job:
                    cf_job.start()
                    cf_job.wait_until_finished()
                try:
                    util.io.fs.remove_recursively(output_dir, not_exist_okay=True)
                except OSError as e:
                    logger.warning('Dir {} could not be removed: {}'.format(output_dir, e))


            ## load cost function values
            if eval_grad_value:
                util.io.matlab.save(df_file, cf.df(parameters), value_name='df', oned_as='column')

            if eval_function_value:
                util.io.matlab.save(f_file, cf.f(parameters), value_name='f', oned_as='column')
