import argparse
import numpy as np

import util.logging
import ndop.model.eval


if __name__ == "__main__":
    with util.logging.Logger():
        parser = argparse.ArgumentParser(description='Starting model spinup.')
        parser.add_argument('p', type=float, nargs=7)
        parser.add_argument('--years', type=int, default=10000)
        parser.add_argument('--tolerance', type=float, default=0.0)
        parser.add_argument('--combination', choices=['or', 'and'], default='or')
        parser.add_argument('--version', action='version', version='%(prog)s 0.1')
        args = vars(parser.parse_args())
        
        p = args['p']
        p = np.array(p)
        years = args['years']
        tolerance = args['tolerance']
        combination = args['combination']
        
#         p = np.array([0.05, 10, 0.05, 0.001, 1, 0, 0.1])
#         years = 1000
#         tolerance=0
#         combination='or'
        
        job_setup = {'name':'NDOP', 'nodes_setup':('f_ocean2', 3, 16)}
        spinup_setup = {'years':years, 'tolerance':tolerance, 'combination':combination}
        model = ndop.model.eval.Model(job_setup)
        model.f_all(p, 12, spinup_setup)