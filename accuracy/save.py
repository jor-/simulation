import argparse
import sys
import numpy as np

from ndop.accuracy.asymptotic import OLS, WLS, GLS
from util.logging import Logger

from ndop.constants import MODEL_OUTPUT_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculating accuracy.')
    
    parser.add_argument('-d', '--kind_of_data', choices=('WOA', 'WOD'), default='WOA', help='The kind of the data to chose.')
    parser.add_argument('-c', '--kind_of_cost_function', choices=('OLS', 'WLS', 'GLS'), default='OLS', help='The kind of the cost function to chose.')
    parser.add_argument('-p', '--parameter_set_nr', type=int, default=184, help='Parameter set nr.')
    parser.add_argument('-t', '--time_dim_df', type=int, default=2880, help='Time dim of df.')
    
    parser.add_argument('-i', '--number_of_measurements', type=int, default=1, help='Number of measurements for increase calulation.')
    
#     parser.add_argument('-s', '--skip_increase', action='store_true', help='Skip increase calulation.')
    parser.add_argument('-m', '--use_mem_map', action='store_true', help='Use memmap to decrease memory use.')
    
    parser.add_argument('-n', '--not_parallel', action='store_false', help='Calculate serial.')
    
    parser.add_argument('-v', '--value_mask_file', help='Calculate average model confidence increase with this value mask.')
    parser.add_argument('-o', '--output_file', help='Save average model confidence increase to this file.')
    
    parser.add_argument('--version', action='version', version='%(prog)s 0.1')
    
    args = vars(parser.parse_args())
    print(args)
    
    data_kind = args['kind_of_data']
    cf_kind = args['kind_of_cost_function']
    parameter_set_nr = args['parameter_set_nr']
    time_dim_df = args['time_dim_df']
    
    number_of_measurements = args['number_of_measurements']
    use_mem_map = args['use_mem_map']
    parallel = args['not_parallel']
    
    output_file = args['output_file']
    value_mask_file = args['value_mask_file']
    if value_mask_file is not None:
        value_mask = np.load(value_mask_file)
    else:
        value_mask = None
#     try:
#         output_file = args['output_file']
#     except KeyError:
#         output_file = None
#     try:
#         value_mask_file = args['value_mask_file']
#         value_mask = np.load(value_mask_file)
#     except KeyError:
#         value_mask = None
    
    if cf_kind == 'OLS':
        cf_class = OLS
    elif cf_kind == 'WLS':
        cf_class = WLS
    elif cf_kind == 'GLS':
        cf_class = GLS
    
    with Logger():
        p = np.loadtxt(MODEL_OUTPUT_DIR+'/time_step_0001/parameter_set_{:0>5}/parameters.txt'.format(parameter_set_nr))        
        spinup_options = spinup_options={'years':10000, 'tolerance':0.0, 'combination':'or'}
#         job_options = {'spinup': {'nodes_setup': ('f_ocean2', 5, 16)}, 'derivative': {'nodes_setup': ('foexpress', 2, 16)}, 'trajectory': {'nodes_setup': ('foexpress', 1, 16)}}
        job_options = {'spinup': {'nodes_setup': ('f_ocean2', 5, 16)}, 'derivative': {'nodes_setup': ('foexpress', 2, 16)}, 'trajectory': {'nodes_setup': ('westmere', 1, 1)}}
        
        asymptotic = cf_class(data_kind, spinup_options, job_setup=job_options)
        asymptotic.parameter_confidence(p)
        asymptotic.model_confidence(p, time_dim_df=time_dim_df, use_mem_map=use_mem_map, parallel_mode=parallel)
        asymptotic.average_model_confidence(p, time_dim_df=time_dim_df, use_mem_map=use_mem_map, parallel_mode=parallel)
        if number_of_measurements > 0:
            average_model_confidence_increase = asymptotic.average_model_confidence_increase(p, number_of_measurements=number_of_measurements, time_dim_df=time_dim_df, value_mask=value_mask, use_mem_map=use_mem_map, parallel_mode=parallel)
            if output_file is not None:
                np.save(output_file, average_model_confidence_increase)
        
        print('finished')
    
    sys.exit()
