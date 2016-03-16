import numpy as np
import os.path

from simulation.constants import METOS3D_DIR
import measurements.land_sea_mask.data


## METOS 3D
METOS_DATA_DIR = os.path.join(METOS3D_DIR, 'data', 'data', 'TMM', '2.8')

METOS_T_RANGE = (0, 1)
METOS_X_RANGE = (0, 360)
METOS_Y_RANGE = (-90, +90)

METOS_T_DIM = 2880
METOS_TIME_STEPS = [2**i for i in range(7)]
LSM = measurements.land_sea_mask.data.LandSeaMaskTMM(t_dim=METOS_T_DIM, t_centered=False)
METOS_SPACE_DIM = LSM.space_dim
METOS_DIM = (METOS_T_DIM,) + METOS_SPACE_DIM
METOS_X_DIM = METOS_SPACE_DIM[0]
METOS_Y_DIM = METOS_SPACE_DIM[1]
METOS_Z_DIM = METOS_SPACE_DIM[2]
METOS_Z_LEFT = LSM.z_left
METOS_Z_CENTER = LSM.z_center


## METOS 3D N-DOP
METOS_SIM_FILE = os.path.join(METOS3D_DIR, 'simpack', 'metos3d-simpack-MITgcm-PO4-DOP.exe')
METOS_TRAJECTORY_FILENAMES = ('sp0000-ts{:0>4}-dop_output.petsc', 'sp0000-ts{:0>4}-po4_output.petsc')
METOS_TRACER_DIM = len(METOS_TRAJECTORY_FILENAMES)


## job
JOB_OPTIONS_FILENAME = 'job_options.hdf5'
JOB_MEMORY_GB = 4


## model spinup
MODEL_SPINUP_MAX_YEARS = 50000
MODEL_START_FROM_CLOSEST_PARAMETER_SET = False
MODEL_DEFAULT_SPINUP_OPTIONS = {'years':10000, 'tolerance':0.0, 'combination':'or'}
# MODEL_DEFAULT_DERIVATIVE_OPTIONS = {'years': 100, 'step_size': 10**(-7), 'accuracy_order': 2}
MODEL_DEFAULT_DERIVATIVE_OPTIONS = {'years': 500, 'step_size': 10**(-6), 'accuracy_order': 2}



## model names
MODEL_NAMES = ['dop_po4',]
MODEL_NAME_TOTAL_CONCENTRATION_SUFFIX = '_c'
MODEL_NAMES = MODEL_NAMES + [model + MODEL_NAME_TOTAL_CONCENTRATION_SUFFIX for model in MODEL_NAMES]


## model parameter
MODEL_PARAMETER_LOWER_BOUND = {'dop_po4': np.array([0, 0, 0, 10**(-8), 10**(-8), 0, 0])}
MODEL_PARAMETER_UPPER_BOUND = {'dop_po4': np.array([METOS_T_DIM, np.inf, 1, np.inf, np.inf, np.inf, np.inf])}
MODEL_PARAMETER_TYPICAL = {'dop_po4': np.array([1, 1, 1, 1, 10, 0.01, 1])}
for model in MODEL_PARAMETER_LOWER_BOUND.keys():
    MODEL_PARAMETER_LOWER_BOUND[model+MODEL_NAME_TOTAL_CONCENTRATION_SUFFIX] = np.concatenate([MODEL_PARAMETER_LOWER_BOUND[model], [0]])
    MODEL_PARAMETER_UPPER_BOUND[model+MODEL_NAME_TOTAL_CONCENTRATION_SUFFIX] = np.concatenate([MODEL_PARAMETER_UPPER_BOUND[model], [np.inf]])
    MODEL_PARAMETER_TYPICAL[model+MODEL_NAME_TOTAL_CONCENTRATION_SUFFIX] = np.concatenate([MODEL_PARAMETER_TYPICAL[model], [1]])


## database directories and files
from simulation.constants import SIMULATION_OUTPUT_DIR as DATABASE_OUTPUT_DIR
DATABASE_MODEL_DIRNAME = 'model_{}'
DATABASE_TIME_STEP_DIRNAME = 'time_step_{:0>4}'
DATABASE_PARAMETERS_SET_DIRNAME = 'parameter_set_{:0>5}'   # substituted by the number of the run to 5 digits
DATABASE_SPINUP_DIRNAME = 'spinup'
DATABASE_DERIVATIVE_DIRNAME = os.path.join('derivative', 'step_size_{}')     # finite differences step size
DATABASE_PARTIAL_DERIVATIVE_DIRNAME = 'partial_derivative_{}_{:+}' # partial_derivative, h_factor
DATABASE_RUN_DIRNAME = 'run_{:0>2}'              # substituted by the number of the run to 2 digits

DATABASE_PARAMETERS_FILENAME = 'parameters.txt'
DATABASE_PARAMETERS_RELIABLE_DECIMAL_PLACES = np.finfo(np.float64).precision
assert DATABASE_PARAMETERS_RELIABLE_DECIMAL_PLACES == 15
DATABASE_PARAMETERS_FORMAT_STRING = '{:.' + '{}'.format(DATABASE_PARAMETERS_RELIABLE_DECIMAL_PLACES) + 'f}'
DATABASE_PARAMETERS_FORMAT_STRING_OLD_STYLE = '%.{}f'.format(DATABASE_PARAMETERS_RELIABLE_DECIMAL_PLACES)

DATABASE_PARAMETERS_LOOKUP_ARRAY_FILENAME = 'database.npy'


## model interpolator
MODEL_INTERPOLATOR_FILE = os.path.join(DATABASE_OUTPUT_DIR, 'interpolator.ppy')
MODEL_INTERPOLATOR_AMOUNT_OF_WRAP_AROUND = (1/METOS_T_DIM, 1/METOS_X_DIM, 0, 0)
MODEL_INTERPOLATOR_NUMBER_OF_LINEAR_INTERPOLATOR = 0
MODEL_INTERPOLATOR_TOTAL_OVERLAPPING_OF_LINEAR_INTERPOLATOR = 0
