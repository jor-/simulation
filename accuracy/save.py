import argparse
import sys
import numpy as np

import ndop.optimization.constants
import ndop.accuracy.asymptotic
from util.logging import Logger

from ndop.constants import MODEL_OUTPUT_DIR
from ndop.optimization.matlab.constants import KIND_OF_COST_FUNCTIONS


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculating accuracy.')

    # parser.add_argument('-d', '--data_kind', choices=('WOA', 'WOD'), default='WOA', help='The kind of the data to chose.')
    # parser.add_argument('-c', '--cf_kind', choices=('OLS',), default='OLS', help='The kind of the cost function to chose.')
    parser.add_argument('-k', '--kind', choices=KIND_OF_COST_FUNCTIONS, help='The kind of the cost function to chose.')
    parser.add_argument('-p', '--parameter_set_nr', type=int, default=184, help='Parameter set nr.')
    parser.add_argument('-t', '--time_dim_df', type=int, default=2880, help='Time dim of df.')

    parser.add_argument('-i', '--number_of_measurements', type=int, default=1, help='Number of measurements for increase calulation.')

    parser.add_argument('-m', '--use_mem_map', action='store_true', help='Use memmap to decrease memory use.')

    parser.add_argument('-n', '--not_parallel', action='store_false', help='Calculate serial.')

    parser.add_argument('-v', '--value_mask_file', default=None, help='Calculate average model confidence increase with this value mask.')
    parser.add_argument('-o', '--output_file', help='Save average model confidence increase to this file.')

    parser.add_argument('--version', action='version', version='%(prog)s 0.1')

    args = parser.parse_args()

    with Logger():
        ## extract infos from kind
        kind_splitted = args.kind.split('.')
        data_kind = kind_splitted[0]
        cf_kind = kind_splitted[1]

        if cf_kind == 'OLS':
            cf_class = ndop.accuracy.asymptotic.OLS
        elif cf_kind == 'WLS':
            cf_class = ndop.accuracy.asymptotic.WLS
        elif cf_kind == 'LWLS':
            cf_class = ndop.accuracy.asymptotic.LWLS
        elif cf_kind == 'GLS':
            cf_class = ndop.accuracy.asymptotic.GLS
            correlation_min_values = int(kind_splitted[2])
            correlation_max_year_diff = int(kind_splitted[3])
            if correlation_max_year_diff < 0:
                correlation_max_year_diff = float('inf')
        else:
            raise ValueError('Unknown cf kind {}.'.format(cf_kind))
        
        ## init asymptotic
        spinup_options = {'years':10000, 'tolerance':0.0, 'combination':'or'}
        job_options = {'spinup': {'nodes_setup': ndop.optimization.constants.COST_FUNCTION_NODES_SETUP_SPINUP}, 'derivative': {'nodes_setup': ndop.optimization.constants.COST_FUNCTION_NODES_SETUP_DERIVATIVE}, 'trajectory': {'nodes_setup': ndop.optimization.constants.COST_FUNCTION_NODES_SETUP_TRAJECTORY}}
        
        cf_kargs = {'data_kind': data_kind, 'spinup_options': spinup_options, 'job_setup': job_options}
        if cf_kind == 'GLS':
            cf_kargs['correlation_min_values'] = correlation_min_values
            cf_kargs['correlation_max_year_diff'] = correlation_max_year_diff
        
        asymptotic = cf_class(**cf_kargs)
        
        ## calculate
        p = np.loadtxt(MODEL_OUTPUT_DIR+'/time_step_0001/parameter_set_{:0>5}/parameters.txt'.format(args.parameter_set_nr))
        asymptotic.parameter_confidence(p)
        asymptotic.model_confidence(p, time_dim_df=args.time_dim_df, use_mem_map=args.use_mem_map, parallel_mode=not args.not_parallel)
        asymptotic.average_model_confidence(p, time_dim_df=args.time_dim_df, use_mem_map=args.use_mem_map, parallel_mode=not args.not_parallel)
        if args.number_of_measurements > 0:
            average_model_confidence_increase = asymptotic.average_model_confidence_increase(p, number_of_measurements=args.number_of_measurements, time_dim_df=args.time_dim_df, value_mask=args.value_mask_file, use_mem_map=args.use_mem_map, parallel_mode=not args.not_parallel)
            if args.output_file is not None:
                np.save(output_file, average_model_confidence_increase)

        print('finished')

    sys.exit()
