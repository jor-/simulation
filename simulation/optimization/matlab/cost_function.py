if __name__ == "__main__":
    
    import argparse
    import sys
    import os
    import tempfile

    import numpy as np
    
    import simulation.model.constants    
    import simulation.model.options
    import simulation.optimization.cost_function
    import simulation.optimization.job
    
    import measurements.all.pw.data
    
    import util.batch.universal.system
    import util.io.matlab
    import util.io.fs
    
    import util.logging
    logger = util.logging.logger
    
    from simulation.optimization.matlab.constants import MATLAB_PARAMETER_FILENAME, MATLAB_F_FILENAME, MATLAB_DF_FILENAME, NODES_MAX_FILENAME, COST_FUNCTION_NAMES
    
    
    ## parse arguments
    parser = argparse.ArgumentParser(description='Evaluating a cost function for matlab.')

    parser.add_argument('--cost_function_name', choices=COST_FUNCTION_NAMES, help='The cost function which should be evaluated.')
    parser.add_argument('--max_box_distance_to_water', type=int, default=float('inf'), help='The maximal distance to water boxes to accept measurements.')
    parser.add_argument('--min_measurements_correlations', type=int, default=float('inf'), help='The minimal number of measurements used to calculate correlations.')
    
    parser.add_argument('--exchange_dir', help='The directory from where to load the parameters and where to save the cost function values.')
    parser.add_argument('--debug_logging_file', default=None, help='File to store debug informations.')

    parser.add_argument('--eval_function_value', action='store_true', help='Save the value of the cost function.')
    parser.add_argument('--eval_grad_value', action='store_true', help='Save the values of the derivative of the cost function.')

    parser.add_argument('--model_name', default=None, choices=simulation.model.constants.MODEL_NAMES, help='The name of the model to use for the simulations.')
    parser.add_argument('--time_step', type=int, default=1, choices=simulation.model.constants.METOS_TIME_STEPS, help='The time step multiplier to use for the simulations.')
    parser.add_argument('--initial_concentrations', type=float, nargs='+', default=None, help='The initial tracer concentrations for the spinup of the model.')
    
    parser.add_argument('--spinup_years', type=int, default=10000, help='The number of years for the spinup.')
    parser.add_argument('--spinup_tolerance', type=float, default=0, help='The tolerance for the spinup.')
    parser.add_argument('--spinup_satisfy_years_and_tolerance', action='store_true', help='If used, the spinup is terminated if years and tolerance have been satisfied. Otherwise, the spinup is terminated as soon as years or tolerance have been satisfied.')

    parser.add_argument('--derivative_step_size', type=float, default=None, help='The step size used for the finite difference approximation.')
    parser.add_argument('--derivative_years', type=int, default=None, help='The number of years for the finite difference approximation spinup.')
    parser.add_argument('--derivative_accuracy_order', type=int, default=None, help='The accuracy order used for the finite difference approximation. 1 = forward differences. 2 = central differences.')

    parser.add_argument('--nodes_setup_node_kind', default=None, help='The node kind to use for the spinup.')
    parser.add_argument('--nodes_setup_number_of_nodes', type=int, default=0, help='The number of nodes to use for the spinup.')
    parser.add_argument('--nodes_setup_number_of_cpus', type=int, default=0, help='The number of cpus to use for the spinup.')

    parser.add_argument('--model_parameters_relative_tolerance', type=float, nargs='+', default=None, help='The relative tolerance up to which two model parameter vectors are considered equal.')
    parser.add_argument('--model_parameters_absolute_tolerance', type=float, nargs='+', default=None, help='The absolute tolerance up to which two model parameter vectors are considered equal.')

    parser.add_argument('--initial_concentrations_relative_tolerance', type=float, default=None, help='The relative tolerance up to which two initial concentration vectors are considered equal.')
    parser.add_argument('--initial_concentrations_absolute_tolerance', type=float, default=None, help='The absolute tolerance up to which two initial concentration vectors are considered equal.')

    parser.add_argument('--version', action='version', version='%(prog)s 0.1')

    args = parser.parse_args()

    ## prepare model options
    model_options = simulation.model.options.ModelOptions()
       
    ## set model name
    if args.model_name is not None:
        model_options['model_name'] = args.model_name 
    
    ## set time step
    model_options['time_step'] = args.time_step
    
    ## set initial concentration
    if args.initial_concentrations is not None:
        model_options['initial_concentration_options'] = {'concentrations': args.initial_concentrations}

    ## set spinup options
    if args.spinup_satisfy_years_and_tolerance:
        combination='and'
    else:
        combination='or'
    model_options['spinup_options'] = {'years': args.spinup_years, 'tolerance': args.spinup_tolerance, 'combination': combination}

    ## set derivative options
    if args.derivative_step_size is not None or args.derivative_years is not None or args.derivative_accuracy_order is not None:
        derivative_options = model_options['derivative_options']
        if args.derivative_step_size is not None:
            derivative_options['step_size'] = args.derivative_step_size
        if args.derivative_years is not None:
            derivative_options['years'] = args.derivative_years
        if args.derivative_accuracy_order is not None:
            derivative_options['accuracy_order'] = args.derivative_accuracy_order
    
    ## set model parameters tolerance options
    if args.model_parameters_relative_tolerance is not None or args.model_parameters_absolute_tolerance is not None:
        parameter_tolerance_options = model_options['parameter_tolerance_options']
        if args.model_parameters_relative_tolerance is not None:
            parameter_tolerance_options['relative'] = np.array(args.model_parameters_relative_tolerance)
        if args.model_parameters_absolute_tolerance is not None:
            parameter_tolerance_options['absolute'] = np.array(args.model_parameters_absolute_tolerance)
    
    ## set initial concentration tolerance options
    if args.initial_concentrations_relative_tolerance is not None or args.initial_concentrations_absolute_tolerance is not None:
        tolerance_options = model_options['initial_concentration_options']['tolerance_options']
        if args.initial_concentrations_relative_tolerance is not None:
            tolerance_options['relative'] = args.model_parameters_relative_tolerance
        if args.initial_concentrations_absolute_tolerance is not None:
            tolerance_options['absolute'] = args.initial_concentrations_absolute_tolerance
    
    ## set job setup
    def prepare_job_options():
        if args.nodes_setup_node_kind is not None:
            from simulation.optimization.constants import COST_FUNCTION_NODES_SETUP_SPINUP
            nodes_setup = COST_FUNCTION_NODES_SETUP_SPINUP.copy()
            nodes_setup['node_kind'] = args.nodes_setup_node_kind
            nodes_setup['nodes'] = args.nodes_setup_number_of_nodes
            nodes_setup['cpus'] = args.nodes_setup_number_of_cpus
            job_options = {'spinup':{'nodes_setup':nodes_setup}}
        else:
            job_options = None
        return job_options

    ## calculate file locations
    exchange_dir = args.exchange_dir
    p_file = os.path.join(exchange_dir, MATLAB_PARAMETER_FILENAME)
    f_file = os.path.join(exchange_dir, MATLAB_F_FILENAME)
    df_file = os.path.join(exchange_dir, MATLAB_DF_FILENAME)
    
    ## load cf parameters
    parameters = util.io.matlab.load(p_file, 'p')

    ## choose cost function
    cf_kind = args.cost_function_name

    if cf_kind == 'OLS':
        cf_class = simulation.optimization.cost_function.OLS
    elif cf_kind == 'WLS':
        cf_class = simulation.optimization.cost_function.WLS
    elif cf_kind == 'LWLS':
        cf_class = simulation.optimization.cost_function.LWLS
    elif cf_kind == 'GLS':
        cf_class = simulation.optimization.cost_function.GLS
    else:
        raise ValueError('Unknown cf kind {}.'.format(cf_kind))

    ## run cost function evaluation
    log_file = args.debug_logging_file
    with util.logging.Logger(log_file=log_file, disp_stdout=log_file is None):
        
        ## choose measurements
        max_box_distance_to_water = args.max_box_distance_to_water
        min_measurements_correlations = args.min_measurements_correlations
        measurements = measurements.all.pw.data.all_measurements(max_box_distance_to_water=max_box_distance_to_water, min_measurements_correlations=min_measurements_correlations, tracers=model_options.tracers)
        
        ## init cost function
        cf = cf_class(measurements_collection=measurements, model_options=model_options, job_options=prepare_job_options())
        cf.parameters = parameters

        ## if necessary start calculation job
        eval_function_value = args.eval_function_value
        eval_grad_value = args.eval_grad_value
        if (eval_function_value and not cf.f_available()) or (eval_grad_value and not cf.df_available()):
            
            ## start spinup job
            cf.model.run_dir

            ## start cf calculation job 
            output_dir = simulation.model.constants.DATABASE_TMP_DIR           
            os.makedirs(output_dir, exist_ok=True)
            output_dir = tempfile.mkdtemp(dir=output_dir, prefix='cost_function_tmp_')
            util.io.fs.add_group_permissions(output_dir)
            
            with simulation.optimization.job.CostFunctionJob(output_dir, cf_kind, model_options, job_options=prepare_job_options(), max_box_distance_to_water=max_box_distance_to_water, min_measurements_correlations=min_measurements_correlations, eval_f=eval_function_value, eval_df=eval_grad_value) as cf_job:
                cf_job.start()
                cf_job.wait_until_finished()
            try:
                util.io.fs.remove_recursively(output_dir, not_exist_okay=True)
            except OSError as e:
                logger.warning('Dir {} could not be removed: {}'.format(output_dir, e))

        ## save cost function values
        if eval_function_value:
            util.io.matlab.save(f_file, cf.f(), value_name='f', oned_as='column')
        if eval_grad_value:
            util.io.matlab.save(df_file, cf.df(), value_name='df', oned_as='column')

