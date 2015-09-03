import os.path

import util.io.env

BASE_DIR = util.io.env.load('NDOP_DIR')

MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, 'model_output')
PARAMETER_OPTIMIZATION_DIR = os.path.join(BASE_DIR, 'parameter_optimization')
OED_DIR = os.path.join(BASE_DIR, 'optimal_experimental_design')
METOS_DIR = os.path.join(BASE_DIR, 'metos3d', 'fixed_version')
