# import os
import os.path

import util.io.env

# NDOP_DIR_ENVIRONMENT_NAME='NDOP_DIR'
# try:
#     BASE_DIR=os.environ[NDOP_DIR_ENVIRONMENT_NAME]
# except KeyError:
#     raise KeyError('The environment variable {} is not set.'.format(NDOP_DIR_ENVIRONMENT_NAME))

BASE_DIR = util.io.env.load('NDOP_DIR')

MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, 'model_output')
PARAMETER_OPTIMIZATION_DIR = os.path.join(BASE_DIR, 'parameter_optimization')
OED_DIR = os.path.join(BASE_DIR, 'optimal_experimental_design')
METOS_DIR = os.path.join(BASE_DIR, 'metos3d', 'fixed_version')
