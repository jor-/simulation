import os.path

import util.io.env


BASE_DIR_ENV_NAME = 'NDOP_DIR'
try:
    BASE_DIR = util.io.env.load(BASE_DIR_ENV_NAME)
    PARAMETER_OPTIMIZATION_DIR = os.path.join(BASE_DIR, 'parameter_optimization')
    OED_DIR = os.path.join(BASE_DIR, 'optimal_experimental_design')
except util.io.env.EnvironmentLookupError:
    pass


SIMULATION_OUTPUT_DIR_ENV_NAME = 'SIMULATION_OUTPUT_DIR'
SIMULATION_OUTPUT_DIR = util.io.env.load(SIMULATION_OUTPUT_DIR_ENV_NAME)

METOS3D_DIR_ENV_NAME = 'METOS3D_DIR'
METOS3D_DIR = util.io.env.load(METOS3D_DIR_ENV_NAME)

