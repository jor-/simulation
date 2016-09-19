import os.path

import numpy as np

import util.constants

import measurements.land_sea_mask.lsm

from simulation.constants import METOS3D_DIR, METOS3D_DIR_ENV_NAME


## METOS 3D
METOS_DATA_DIR = os.path.join(METOS3D_DIR, 'data', 'data', 'TMM', '2.8')
METOS_DATA_DIR_ENV = os.path.join('${{{}}}'.format(METOS3D_DIR_ENV_NAME), 'data', 'data', 'TMM', '2.8')
METOS_SIM_FILE = os.path.join(METOS3D_DIR, 'metos3d', 'metos3d-simpack-{model_name}.exe')
METOS_SIM_FILE_ENV = os.path.join('${{{}}}'.format(METOS3D_DIR_ENV_NAME), 'metos3d', 'metos3d-simpack-{model_name}.exe')

METOS_T_RANGE = (0, 1)
METOS_X_RANGE = (0, 360)
METOS_Y_RANGE = (-90, +90)

METOS_T_DIM = 2880
METOS_TIME_STEPS = [2**i for i in range(7)]
METOS_LSM = measurements.land_sea_mask.lsm.LandSeaMaskTMM(t_dim=METOS_T_DIM, t_centered=False)
METOS_SPACE_DIM = METOS_LSM.space_dim
METOS_DIM = (METOS_T_DIM,) + METOS_SPACE_DIM
METOS_X_DIM = METOS_SPACE_DIM[0]
METOS_Y_DIM = METOS_SPACE_DIM[1]
METOS_Z_DIM = METOS_SPACE_DIM[2]
METOS_Z_LEFT = METOS_LSM.z_left
METOS_Z_CENTER = METOS_LSM.z_center
METOS_VECTOR_LEN = 52749


METOS_TRAJECTORY_FILENAME = 'sp0000-ts{time_step:0>4d}-{tracer}_output.petsc'

## METOS 3D N-DOP
METOS_TRAJECTORY_FILENAMES = ('sp0000-ts{:0>4}-dop_output.petsc', 'sp0000-ts{:0>4}-po4_output.petsc')
METOS_TRACER_DIM = len(METOS_TRAJECTORY_FILENAMES)


## job
JOB_OPTIONS_FILENAME = 'job_options.hdf5'
JOB_MEMORY_GB = 4


## model spinup
MODEL_SPINUP_MAX_YEARS = 50000
MODEL_START_FROM_CLOSEST_PARAMETER_SET = False
MODEL_DEFAULT_SPINUP_OPTIONS = {'years':10000, 'tolerance':0.0, 'combination':'or'}
MODEL_DEFAULT_DERIVATIVE_OPTIONS = {'years': 500, 'step_size': 10**(-6), 'accuracy_order': 2}


## model names
MODEL_NAMES = ['MITgcm-PO4-DOP', 'N', 'N-DOP', 'NP-DOP', 'NPZ-DOP', 'NPZD-DOP']


## model tracer 
MODEL_TRACER = {}
MODEL_TRACER['MITgcm-PO4-DOP'] = ('po4', 'dop')
MODEL_TRACER['N'] = ('po4',)
MODEL_TRACER['N-DOP'] = ('po4', 'dop')
MODEL_TRACER['NP-DOP'] = ('po4', 'p', 'dop')
MODEL_TRACER['NPZ-DOP'] = ('po4', 'p', 'z', 'dop')
MODEL_TRACER['NPZD-DOP'] = ('po4', 'p', 'z', 'd', 'dop')
assert len(MODEL_TRACER['MITgcm-PO4-DOP']) == 2
assert len(MODEL_TRACER['N']) == 1
assert len(MODEL_TRACER['N-DOP']) == 2
assert len(MODEL_TRACER['NP-DOP']) == 3
assert len(MODEL_TRACER['NPZ-DOP']) == 4
assert len(MODEL_TRACER['NPZD-DOP']) == 5


## model inital concentrations 
MODEL_DEFAULT_INITIAL_CONCENTRATION = {}
MODEL_DEFAULT_INITIAL_CONCENTRATION['MITgcm-PO4-DOP'] = (2.17, 10**-4)
MODEL_DEFAULT_INITIAL_CONCENTRATION['N'] = (2.17,)
MODEL_DEFAULT_INITIAL_CONCENTRATION['N-DOP'] = (2.17, 10**-4)
MODEL_DEFAULT_INITIAL_CONCENTRATION['NP-DOP'] = (2.17, 10**-4, 10**-4)
MODEL_DEFAULT_INITIAL_CONCENTRATION['NPZ-DOP'] = (2.17, 10**-4, 10**-4, 10**-4)
MODEL_DEFAULT_INITIAL_CONCENTRATION['NPZD-DOP'] = (2.17, 10**-4, 10**-4, 10**-4, 10**-4)
assert len(MODEL_DEFAULT_INITIAL_CONCENTRATION['MITgcm-PO4-DOP']) == len(MODEL_TRACER['MITgcm-PO4-DOP'])
assert len(MODEL_DEFAULT_INITIAL_CONCENTRATION['N']) == len(MODEL_TRACER['N'])
assert len(MODEL_DEFAULT_INITIAL_CONCENTRATION['N-DOP']) == len(MODEL_TRACER['N-DOP'])
assert len(MODEL_DEFAULT_INITIAL_CONCENTRATION['NP-DOP']) == len(MODEL_TRACER['NP-DOP'])
assert len(MODEL_DEFAULT_INITIAL_CONCENTRATION['NPZ-DOP']) == len(MODEL_TRACER['NPZ-DOP'])
assert len(MODEL_DEFAULT_INITIAL_CONCENTRATION['NPZD-DOP']) == len(MODEL_TRACER['NPZD-DOP'])


## model parameter
MODEL_PARAMETER_BOUNDS = {}

MODEL_PARAMETER_BOUNDS['MITgcm-PO4-DOP'] = np.array([
[0, METOS_T_DIM],       # lambdaDOPprime  = u(1)/360.d0   ! DOP reminalization rate   [1/y]
[0, np.inf],            # muP             = u(2)          ! maximum groth rate P      [1/d]
[0, 1],                 # sigmaDOP        = u(3)          ! fraction of DOP           [1]
[10**(-8), np.inf],     # KN              = u(4)          ! N half saturation         [mmolP/m^3]
[10**(-8), np.inf],     # KI              = u(5)          ! I half saturation         [W/m^2]
[0, np.inf],            # kw              = u(6)          ! attenuation of water      [1/m]
[0, np.inf]             # b               = u(7)          ! power law coefficient     [1]
])
assert MODEL_PARAMETER_BOUNDS['MITgcm-PO4-DOP'].shape == (7, 2)

MODEL_PARAMETER_BOUNDS['N'] = np.array([
[0, np.inf],            # kw  = u(1)          ! attenuation of water      [1/m]
[0, np.inf],            # muP = u(2)          ! maximum groth rate P      [1/d]
[10**(-8), np.inf],     # KN  = u(3)          ! N half saturation         [mmolP/m^3]
[10**(-8), np.inf],     # KI  = u(4)          ! I half saturation         [W/m^2]
[0, np.inf]             # b   = u(5)          ! power law coefficient     [1]
])
assert MODEL_PARAMETER_BOUNDS['N'].shape == (5, 2)

MODEL_PARAMETER_BOUNDS['N-DOP'] = np.array([
[0, np.inf],            # kw              = u(1)          ! attenuation of water      [1/m]
[0, np.inf],            # muP             = u(2)          ! maximum groth rate P      [1/d]
[10**(-8), 1],          # KN              = u(3)          ! N half saturation         [mmolP/m^3]
[10**(-8), np.inf],     # KI              = u(4)          ! I half saturation         [W/m^2]
[0, 1],                 # sigmaDOP        = u(5)          ! fraction of DOP           [1]
[0, METOS_T_DIM],       # lambdaDOPprime  = u(6)/360.d0   ! DOP reminalization rate   [1/y]
[0, np.inf]             # b               = u(7)          ! power law coefficient     [1]
])
assert MODEL_PARAMETER_BOUNDS['N-DOP'].shape == (7, 2)

MODEL_PARAMETER_BOUNDS['NP-DOP'] = np.array([
[0, np.inf],            # kw              = u(1)          ! attenuation of water                  [1/m]
[0, np.inf],            # kc              = u(2)          ! attenuation of chlorophyll (P)        [1/m (m^3/mmolP)]
[0, np.inf],            # muP             = u(3)          ! maximum groth rate P                  [1/d]
[0, np.inf],            # muZ             = u(4)          ! maximum groth rate Z                  [1/d]
[10**(-8), np.inf],     # KN              = u(5)          ! N half saturation                     [mmolP/m^3]
[10**(-8), np.inf],     # KP              = u(6)          ! P half saturation                     [mmolP/m^3]
[10**(-8), np.inf],     # KI              = u(7)          ! I half saturation                     [W/m^2]
[0, 1],                 # sigmaDOP        = u(8)          ! fraction of DOP                       [1]
[0, np.inf],            # lambdaP         = u(9)          ! linear loss rate P (euphotic)         [1/d]
[0, np.inf],            # kappaP          = u(10)         ! quadratic loss rate P (euphotic)      [1/d (m^3/mmolP)]
[0, np.inf],            # lambdaPprime    = u(11)         ! linear loss rate Z (all layers)       [1/d]
[0, METOS_T_DIM],       # lambdaDOPprime  = u(12)/360.d0  ! DOP reminalization rate (all layers)  [1/y]
[0, np.inf],            # b               = u(13)         ! power law coefficient                 [1]
])
assert MODEL_PARAMETER_BOUNDS['NP-DOP'].shape == (13, 2)

MODEL_PARAMETER_BOUNDS['NPZ-DOP'] = np.array([
[0, np.inf],            # kw              = u(1)          ! attenuation of water                  [1/m]
[0, np.inf],            # kc              = u(2)          ! attenuation of chlorophyll (P)        [1/m (m^3/mmolP)]
[0, np.inf],            # muP             = u(3)          ! maximum groth rate P                  [1/d]
[0, np.inf],            # muZ             = u(4)          ! maximum groth rate Z                  [1/d]
[10**(-8), np.inf],     # KN              = u(5)          ! N half saturation                     [mmolP/m^3]
[10**(-8), np.inf],     # KP              = u(6)          ! P half saturation                     [mmolP/m^3]
[10**(-8), np.inf],     # KI              = u(7)          ! I half saturation                     [W/m^2]
[0, 1],                 # sigmaZ          = u(8)          ! fraction of Z                         [1]
[0, 1],                 # sigmaDOP        = u(9)          ! fraction of DOP                       [1]
[0, np.inf],            # lambdaP         = u(10)         ! linear loss rate P (euphotic)         [1/d]
[0, np.inf],            # lambdaZ         = u(11)         ! linear loss rate Z (euphotic)         [1/d]
[0, np.inf],            # kappaZ          = u(12)         ! quadratic loss rate Z (euphotic)      [1/d (m^3/mmolP)]
[0, np.inf],            # lambdaPprime    = u(13)         ! linear loss rate P (all layers)       [1/d]
[0, np.inf],            # lambdaZprime    = u(14)         ! linear loss rate Z (all layers)       [1/d]
[0, METOS_T_DIM],       # lambdaDOPprime  = u(15)/360.d0  ! DOP reminalization rate (all layers)  [1/y]
[0, np.inf],            # b               = u(16)         ! power law coefficient                 [1]
])
assert MODEL_PARAMETER_BOUNDS['NPZ-DOP'].shape == (16, 2)

MODEL_PARAMETER_BOUNDS['NPZD-DOP'] = np.array([
[0, np.inf],            # kw              = u(1)          ! attenuation of water                  [1/m]
[0, np.inf],            # kc              = u(2)          ! attenuation of chlorophyll (P)        [1/m (m^3/mmolP)]
[0, np.inf],            # muP             = u(3)          ! maximum groth rate P                  [1/d]
[0, np.inf],            # muZ             = u(4)          ! maximum groth rate Z                  [1/d]
[10**(-8), np.inf],     # KN              = u(5)          ! N half saturation                     [mmolP/m^3]
[10**(-8), np.inf],     # KP              = u(6)          ! P half saturation                     [mmolP/m^3]
[10**(-8), np.inf],     # KI              = u(7)          ! I half saturation                     [W/m^2]
[0, 1],                 # sigmaZ          = u(8)          ! fraction of Z                         [1]
[0, 1],                 # sigmaDOP        = u(9)          ! fraction of DOP                       [1]
[0, np.inf],            # lambdaP         = u(10)         ! linear loss rate P (euphotic)         [1/d]
[0, np.inf],            # lambdaZ         = u(11)         ! linear loss rate Z (euphotic)         [1/d]
[0, np.inf],            # kappaZ          = u(12)         ! quadratic loss rate Z (euphotic)      [1/d (m^3/mmolP)]
[0, np.inf],            # lambdaPprime    = u(13)         ! linear loss rate P (all layers)       [1/d]
[0, np.inf],            # lambdaZprime    = u(14)         ! linear loss rate Z (all layers)       [1/d]
[0, np.inf],            # lambdaDprime    = u(15)         ! linear lass rate D (all layers)           [1/d]
[0, METOS_T_DIM],       # lambdaDOPprime  = u(16)/360.d0  ! DOP reminalization rate (all layers)  [1/y]
[0, np.inf],            # aD              = u(17)         ! increase of sinking speed w.r.t. depth    [1/d]
[0, np.inf],            # bD              = u(18)         ! initial sinking speed                     [m/d]
])
assert MODEL_PARAMETER_BOUNDS['NPZD-DOP'].shape == (18, 2)


MODEL_PARAMETER_TYPICAL = {}
MODEL_PARAMETER_TYPICAL['MITgcm-PO4-DOP'] = np.array([1, 1, 1, 1, 10, 0.01, 1])
MODEL_PARAMETER_TYPICAL['N'] = np.array([0.01, 1, 1, 10, 1])
MODEL_PARAMETER_TYPICAL['N-DOP'] = np.array([0.01, 1, 1, 10, 1, 1, 1])
MODEL_PARAMETER_TYPICAL['NP-DOP'] = np.array([0.01, 1, 1, 1, 1, 1, 10, 1, 0.01, 1, 0.01, 1, 1])
MODEL_PARAMETER_TYPICAL['NPZ-DOP'] = np.array([0.01, 1, 1, 1, 1, 1, 10, 1, 1, 0.01, 0.01, 1, 0.01, 0.01, 1, 1])
MODEL_PARAMETER_TYPICAL['NPZD-DOP'] = np.array([0.01, 1, 1, 1, 1, 1, 10, 1, 1, 0.01, 0.01, 1, 0.01, 0.01, 0.01, 1, 0.01, 0.01])

assert len(MODEL_PARAMETER_TYPICAL['MITgcm-PO4-DOP']) == len(MODEL_PARAMETER_BOUNDS['MITgcm-PO4-DOP'])
assert len(MODEL_PARAMETER_TYPICAL['N']) ==  len(MODEL_PARAMETER_BOUNDS['N'])
assert len(MODEL_PARAMETER_TYPICAL['N-DOP']) ==  len(MODEL_PARAMETER_BOUNDS['N-DOP'])
assert len(MODEL_PARAMETER_TYPICAL['NP-DOP']) ==  len(MODEL_PARAMETER_BOUNDS['NP-DOP'])
assert len(MODEL_PARAMETER_TYPICAL['NPZ-DOP']) ==  len(MODEL_PARAMETER_BOUNDS['NPZ-DOP'])
assert len(MODEL_PARAMETER_TYPICAL['NPZD-DOP']) ==  len(MODEL_PARAMETER_BOUNDS['NPZD-DOP'])



## database directories and files
from simulation.constants import SIMULATION_OUTPUT_DIR as DATABASE_OUTPUT_DIR
DATABASE_MODEL_DIRNAME = 'model_{}'
DATABASE_TIME_STEP_DIRNAME = 'time_step_{:0>4d}'
DATABASE_SPINUP_DIRNAME = 'spinup'
DATABASE_DERIVATIVE_DIRNAME = os.path.join('derivative', 'step_size_{step_size}')
DATABASE_PARTIAL_DERIVATIVE_DIRNAME = 'partial_derivative_{kind}_{index:d}_{h_factor:+d}'
DATABASE_RUN_DIRNAME = 'run_{:0>5d}'

DATABASE_VECTOR_CONCENTRATIONS_DIRNAME = 'initial_concentration_vector'
DATABASE_VECTOR_CONCENTRATIONS_FILENAME = 'concentration_{tracer}.petsc'
DATABASE_VECTOR_CONCENTRATIONS_RELIABLE_DECIMAL_PLACES = np.finfo(np.float64).precision

DATABASE_CONSTANT_CONCENTRATIONS_DIRNAME = 'initial_concentration_constant'
DATABASE_CONSTANT_CONCENTRATIONS_LOOKUP_ARRAY_FILENAME = 'concentrations_database.npy'
DATABASE_CONSTANT_CONCENTRATIONS_FILENAME = 'constant_concentrations.txt'
DATABASE_CONSTANT_CONCENTRATIONS_RELIABLE_DECIMAL_PLACES = np.finfo(np.float64).precision

DATABASE_CONCENTRATIONS_DIRNAME = 'concentration_{:0>5d}'

DATABASE_PARAMETERS_DIRNAME = 'parameter_set_{:0>5d}'
DATABASE_PARAMETERS_FILENAME = 'parameters.txt'
DATABASE_PARAMETERS_LOOKUP_ARRAY_FILENAME = 'parameter_set_database.npy'
DATABASE_PARAMETERS_RELIABLE_DECIMAL_PLACES = np.finfo(np.float64).precision
assert DATABASE_PARAMETERS_RELIABLE_DECIMAL_PLACES == 15
DATABASE_PARAMETERS_FORMAT_STRING = '{:.' + '{}'.format(DATABASE_PARAMETERS_RELIABLE_DECIMAL_PLACES) + 'f}'


DATABASE_POINTS_OUTPUT_DIRNAME = os.path.join('output', '{tracer}', '{data_set_name}')
DATABASE_ALL_DATASET_NAME = 'all_model_values_-_time_dim_{time_dim}'
DATABASE_F_FILENAME = 'f.npz'
DATABASE_DF_FILENAME = 'df_{derivative_kind}.npz'
DATABASE_CACHE_OPTION_FILE_SUFFIX = '_options'

DATABASE_TMP_DIR = os.path.join(util.constants.TMP_DIR, 'metos3d_simulations')


## model interpolator
MODEL_INTERPOLATOR_FILE = os.path.join(DATABASE_OUTPUT_DIR, 'interpolator.ppy')
MODEL_INTERPOLATOR_AMOUNT_OF_WRAP_AROUND = (1/METOS_T_DIM, 1/METOS_X_DIM, 0, 0)
MODEL_INTERPOLATOR_NUMBER_OF_LINEAR_INTERPOLATOR = 0
MODEL_INTERPOLATOR_SINGLE_OVERLAPPING_AMOUNT_OF_LINEAR_INTERPOLATOR = 0
