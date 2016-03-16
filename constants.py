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
try:
    SIMULATION_OUTPUT_DIR = util.io.env.load(SIMULATION_OUTPUT_DIR_ENV_NAME)
except util.io.env.EnvironmentLookupError:
    SIMULATION_OUTPUT_DIR = os.path.join(BASE_DIR, 'model_output')

METOS3D_DIR_ENV_NAME = 'METOS3D_DIR'
try:
    METOS3D_DIR = util.io.env.load(METOS3D_DIR_ENV_NAME)
except util.io.env.EnvironmentLookupError:
    METOS3D_DIR = os.path.join(BASE_DIR, 'metos3d', 'fixed_version')


