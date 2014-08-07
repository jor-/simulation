import numpy as np
import os

from ndop.constants import BASE_DIR, MODEL_OUTPUT_DIR


JOB_OPTIONS_FILENAME = 'job_options.hdf5'
JOB_MEMORY_GB = 4
JOB_MIN_CPUS = 32

MODEL_TIME_STEP_SIZE_MAX = 2880
MODEL_TIME_STEP_DIRNAME = 'time_step_{:0>4}'

MODEL_PARAMETER_DIM = 7
MODEL_PARAMETER_LOWER_BOUND = np.array([0, 0, 0, 0, 0, 0, 0])
MODEL_PARAMETER_UPPER_BOUND = np.array([1, np.inf, 1, np.inf, np.inf, np.inf, np.inf])
MODEL_PARAMETERS_MAX_DIFF = 10**(-10) * MODEL_PARAMETER_DIM

MODEL_PARAMETERS_SET_DIRNAME = 'parameter_set_{:0>5}'   # substituted by the number of the run to 5 digits
MODEL_PARAMETERS_FILENAME = 'parameters.txt'
MODEL_PARAMETERS_FORMAT_STRING = '%.18f'

MODEL_SPINUP_DIRNAME = 'spinup'
MODEL_DERIVATIVE_DIRNAME = 'derivative'
MODEL_PARTIAL_DERIVATIVE_DIRNAME = 'partial_derivative_{}_{:+}' # partial_derivative, h_factor
MODEL_RUN_DIRNAME = 'run_{:0>2}'              # substituted by the number of the run to 2 digits
MODEL_RUN_OPTIONS_FILENAME = 'run_options.txt'

# MODEL_TMP_DIR = os.environ['TMPDIR']
MODEL_TMP_DIR = None

MODEL_SPINUP_MAX_YEARS = 50000
MODEL_DERIVATIVE_SPINUP_YEARS = 100
MODEL_START_FROM_CLOSEST_PARAMETER_SET = False


MODEL_INTERPOLATOR_FILE = os.path.join(MODEL_OUTPUT_DIR, 'interpolator.ppy')
MODEL_INTERPOLATOR_AMOUNT_OF_WRAP_AROUND = 1/2880
MODEL_INTERPOLATOR_NUMBER_OF_LINEAR_INTERPOLATOR = 0
MODEL_INTERPOLATOR_TOTAL_OVERLAPPING_OF_LINEAR_INTERPOLATOR = 0


METOS_PATH_1 = os.path.join(BASE_DIR, 'metos3d/v0.2')
METOS_PATH_2 = os.path.join(BASE_DIR, 'metos3d/v_rz_2')

METOS_TRAJECTORY_FILENAMES = ('sp0000-ts{:0>4}-dop_output.petsc', 'sp0000-ts{:0>4}-po4_output.petsc')
METOS_TRACER_DIM = len(METOS_TRAJECTORY_FILENAMES)

METOS_T_RANGE = (0, 1)
METOS_X_RANGE = (0, 360)
METOS_Y_RANGE = (-90, +90)
from measurements.land_sea_mask.constants import LSM_128x64x15_Z_LEFT as METOS_Z_LEFT
from measurements.land_sea_mask.constants import LSM_128x64x15_Z_CENTER as METOS_Z_CENTER
from measurements.land_sea_mask.constants import LSM_128x64x15_DIM as METOS_DIM
METOS_X_DIM = METOS_DIM[0]
METOS_Y_DIM = METOS_DIM[1]
METOS_Z_DIM = METOS_DIM[2]

