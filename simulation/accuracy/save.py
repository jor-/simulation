if __name__ == "__main__":
    
    import argparse
    import sys
    import numpy as np
    
    import simulation.optimization.constants
    import simulation.accuracy.asymptotic
    
    import util.parallel.universal
    from util.logging import Logger
    
    from simulation.constants import SIMULATION_OUTPUT_DIR
    from simulation.optimization.matlab.constants import KIND_OF_COST_FUNCTIONS


    parser = argparse.ArgumentParser(description='Calculating accuracy.')

    parser.add_argument('-k', '--kind', choices=KIND_OF_COST_FUNCTIONS, help='The kind of the cost function to chose.')
    parser.add_argument('-p', '--parameter_set_nr', type=int, default=184, help='Parameter set nr.')
    parser.add_argument('-t', '--time_dim_df', type=int, default=2880, help='Time dim of df.')
    parser.add_argument('-c', '--time_dim_confidence_increase', type=int, default=12, help='Time dim of confidence increase.')

    parser.add_argument('-i', '--number_of_measurements', type=int, default=1, help='Number of measurements for increase calulation.')

    parser.add_argument('-m', '--use_mem_map', action='store_true', help='Use memmap to decrease memory use.')

    parser.add_argument('-n', '--not_parallel', action='store_false', help='Calculate serial.')

    parser.add_argument('-v', '--value_mask_file', default=None, help='Calculate average model confidence increase with this value mask.')
    parser.add_argument('-o', '--output_file', help='Save average model confidence increase to this file.')

    parser.add_argument('--version', action='version', version='%(prog)s 0.1')

    args = parser.parse_args()

    with Logger():
        ## extract infos from kind
        kind_splitted = args.kind.split('_')
        assert len(kind_splitted) == 2
        data_kind = kind_splitted[0]
        cf_kind = kind_splitted[1]
        time_step = 1
        
        from simulation.optimization.constants import COST_FUNCTION_NODES_SETUP_SPINUP, COST_FUNCTION_NODES_SETUP_DERIVATIVE, COST_FUNCTION_NODES_SETUP_TRAJECTORY
        job_setup = {'name':'Accuracy'}
        job_setup['spinup'] = {'nodes_setup' : COST_FUNCTION_NODES_SETUP_SPINUP}
        job_setup['derivative'] = {'nodes_setup' : COST_FUNCTION_NODES_SETUP_DERIVATIVE}
        job_setup['trajectory'] = {'nodes_setup' : COST_FUNCTION_NODES_SETUP_TRAJECTORY}
        
        asymptotic_kargs = {'data_kind': data_kind, 'model_options': {'time_step': time_step, 'total_concentration_factor_included_in_parameters': True}, 'job_setup': job_setup}

        if cf_kind == 'OLS':
            asymptotic_class = simulation.accuracy.asymptotic.OLS
        elif cf_kind == 'WLS':
            asymptotic_class = simulation.accuracy.asymptotic.WLS
        elif cf_kind == 'LWLS':
            asymptotic_class = simulation.accuracy.asymptotic.LWLS
        elif cf_kind.startswith('GLS'):
            asymptotic_class = simulation.accuracy.asymptotic.GLS
            cf_kind_splitted = cf_kind.split('.')
            correlation_min_values = int(cf_kind_splitted[1])
            correlation_max_year_diff = int(cf_kind_splitted[2])
            if correlation_max_year_diff < 0:
                correlation_max_year_diff = float('inf')
            asymptotic_kargs['correlation_min_values'] = correlation_min_values
            asymptotic_kargs['correlation_max_year_diff'] = correlation_max_year_diff
        else:
            raise ValueError('Unknown cf kind {}.'.format(cf_kind))
        
        ## init asymptotic
        asymptotic = asymptotic_class(**asymptotic_kargs)
        
        ## parallel mode
        if not args.not_parallel:
            parallel_mode = util.parallel.universal.MODES['serial']
        else:
            parallel_mode = util.parallel.universal.max_parallel_mode()
        
        ## calculate
        p = np.loadtxt(SIMULATION_OUTPUT_DIR+'/model_dop_po4/time_step_0001/parameter_set_{:0>5}/parameters.txt'.format(args.parameter_set_nr))
        asymptotic.parameter_confidence(p)
        asymptotic.model_confidence(p, time_dim_df=args.time_dim_df, use_mem_map=args.use_mem_map, parallel_mode=parallel_mode)
        asymptotic.average_model_confidence(p, time_dim_df=args.time_dim_df, use_mem_map=args.use_mem_map, parallel_mode=parallel_mode)
        if args.number_of_measurements > 0:
            if args.value_mask_file is not None:
                value_mask = np.load(args.value_mask_file)
            else:
                value_mask = None
            average_model_confidence_increase = asymptotic.average_model_confidence_increase(p, number_of_measurements=args.number_of_measurements, time_dim_confidence_increase=args.time_dim_confidence_increase, time_dim_df=args.time_dim_df, value_mask=value_mask, use_mem_map=args.use_mem_map, parallel_mode=parallel_mode)
            if args.output_file is not None:
                np.save(args.output_file, average_model_confidence_increase)

        print('finished')

    sys.exit()
