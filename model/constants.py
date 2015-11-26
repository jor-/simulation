import numpy as np
import os.path

# from ndop.constants import BASE_DIR, MODEL_OUTPUT_DIR
from ndop.constants import MODEL_OUTPUT_DIR, METOS_DIR
import measurements.land_sea_mask.data


## METOS 3D
METOS_DATA_DIR = os.path.join(METOS_DIR, 'data', 'data', 'TMM', '2.8')
METOS_SIM_FILE = os.path.join(METOS_DIR, 'simpack', 'metos3d-simpack-MITgcm-PO4-DOP.exe')


## METOS 3D N-DOP
METOS_TRAJECTORY_FILENAMES = ('sp0000-ts{:0>4}-dop_output.petsc', 'sp0000-ts{:0>4}-po4_output.petsc')
METOS_TRACER_DIM = len(METOS_TRAJECTORY_FILENAMES)

METOS_T_RANGE = (0, 1)
METOS_X_RANGE = (0, 360)
METOS_Y_RANGE = (-90, +90)
METOS_T_DIM = 2880
LSM = measurements.land_sea_mask.data.LandSeaMaskTMM(t_dim=METOS_T_DIM, t_centered=False)
METOS_SPACE_DIM = LSM.space_dim
METOS_DIM = (METOS_T_DIM,) + METOS_SPACE_DIM
METOS_X_DIM = METOS_SPACE_DIM[0]
METOS_Y_DIM = METOS_SPACE_DIM[1]
METOS_Z_DIM = METOS_SPACE_DIM[2]
METOS_Z_LEFT = LSM.z_left
METOS_Z_CENTER = LSM.z_center


## Job
JOB_OPTIONS_FILENAME = 'job_options.hdf5'
JOB_MEMORY_GB = 4
JOB_MIN_CPUS = 32


## Model parameter
MODEL_PARAMETER_DIM = 7
MODEL_PARAMETER_LOWER_BOUND = np.array([0, 0, 0, 10**(-8), 10**(-8), 0, 0])
MODEL_PARAMETER_UPPER_BOUND = np.array([METOS_T_DIM, np.inf, 1, np.inf, np.inf, np.inf, np.inf])
MODEL_PARAMETER_TYPICAL = np.array([1, 1, 1, 1, 10, 0.01, 1])
# MODEL_PARAMETERS_MAX_DIFF = 10**(-10) * MODEL_PARAMETER_DIM
MODEL_PARAMETERS_MAX_REL_DIFF = 10**(-5)
MODEL_PARAMETERS_MAX_ABS_DIFF = 10**(-8)


## Model directories and files
MODEL_TIME_STEP_DIRNAME = 'time_step_{:0>4}'
MODEL_PARAMETERS_SET_DIRNAME = 'parameter_set_{:0>5}'   # substituted by the number of the run to 5 digits
MODEL_SPINUP_DIRNAME = 'spinup'
MODEL_DERIVATIVE_DIRNAME = os.path.join('derivative', 'step_size_{}')     # finite differences step size
MODEL_PARTIAL_DERIVATIVE_DIRNAME = 'partial_derivative_{}_{:+}' # partial_derivative, h_factor
MODEL_RUN_DIRNAME = 'run_{:0>2}'              # substituted by the number of the run to 2 digits

MODEL_RUN_OPTIONS_FILENAME = 'run_options.txt'

MODEL_PARAMETERS_FILENAME = 'parameters.txt'
MODEL_PARAMETERS_RELIABLE_DECIMAL_PLACES = 18
MODEL_PARAMETERS_FORMAT_STRING = '{:.' + '{}'.format(MODEL_PARAMETERS_RELIABLE_DECIMAL_PLACES) + 'f}'
MODEL_PARAMETERS_FORMAT_STRING_OLD_STYLE = '%.{}f'.format(MODEL_PARAMETERS_RELIABLE_DECIMAL_PLACES)

MODEL_TMP_DIR = os.path.join(MODEL_OUTPUT_DIR, 'tmp')

## Model spinup
MODEL_SPINUP_MAX_YEARS = 50000
MODEL_START_FROM_CLOSEST_PARAMETER_SET = False
MODEL_DEFAULT_SPINUP_OPTIONS = {'years':10000, 'tolerance':0.0, 'combination':'or'}
MODEL_DEFAULT_DERIVATIVE_OPTIONS = {'years': 100, 'step_size': 10**(-7), 'accuracy_order': 2}

## Model interpolator
MODEL_INTERPOLATOR_FILE = os.path.join(MODEL_OUTPUT_DIR, 'interpolator.ppy')
MODEL_INTERPOLATOR_AMOUNT_OF_WRAP_AROUND = (1/METOS_T_DIM, 1/METOS_X_DIM, 0, 0)
MODEL_INTERPOLATOR_NUMBER_OF_LINEAR_INTERPOLATOR = 0
MODEL_INTERPOLATOR_TOTAL_OVERLAPPING_OF_LINEAR_INTERPOLATOR = 0

