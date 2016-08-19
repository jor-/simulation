import argparse
import sys
import numpy as np

from simulation.oed.cost_function import CostFunction
from util.logging import Logger

from simulation.constants import SIMULATION_OUTPUT_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate optimal points.')

    parser.add_argument('-p', '--parameter_set_nr', type=int, default=184, help='Parameter set nr.')
    parser.add_argument('-t', '--time_dim_df', type=int, default=2880, help='Time dim of df.')
    parser.add_argument('-n', '--number_of_measurements', type=int, default=1, help='Number of measurements.')
    parser.add_argument('-v', '--value_mask_file', help='Use this value mask.')
    parser.add_argument('-o', '--output_file', help='Save to this file.')

    parser.add_argument('-d', '--debug', action='store_true', help='Print debug infos.')
    parser.add_argument('--version', action='version', version='%(prog)s 0.1')

    parser.add_argument('-i', '--initial_individuals', type=int, default=100, help='Number of initial individuals.')
    parser.add_argument('-g', '--generations', type=int, default=50, help='Number of generations.')

    args = vars(parser.parse_args())
    print(args)

    parameter_set_nr = args['parameter_set_nr']
    time_dim_df = args['time_dim_df']
    number_of_measurements = args['number_of_measurements']
    value_mask_file = args['value_mask_file']
    output_file = args['output_file']
    debug = args['debug']
    if value_mask_file is not None:
        value_mask = np.load(value_mask_file)
    else:
        value_mask = None
    initial_individuals = args['initial_individuals']
    generations = args['generations']


    with Logger(debug):
        p = np.loadtxt(SIMULATION_OUTPUT_DIR+'/model_dop_po4/time_step_0001/parameter_set_{:0>5}/parameters.txt'.format(parameter_set_nr))
        cf = CostFunction(p, time_dim_df=time_dim_df, value_mask=value_mask)
        p_opt = cf.optimize(number_of_measurements, number_of_initial_individuals=initial_individuals, number_of_generations=generations)
        np.save(output_file, p_opt)

    print('FINISHED')
    sys.exit()
