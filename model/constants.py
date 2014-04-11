import numpy as np
import os.path

from ndop.constants import BASE_DIR, MODEL_OUTPUT_DIR


JOB_OPTIONS_FILENAME = 'job_options.hdf5'
JOB_MEMORY_GB = 4
JOB_MIN_CPUS = 3*12
JOB_NODES_MAX = (12, 13, 12, 24, 3, 1, 0, 0)
JOB_NODES_LEFT_FREE = (0, 0, 0, 24, 3, 1, 0, 0)


MODEL_TIME_STEP_SIZE_MAX = 2880
# MODEL_TIME_STEP_DIRNAME = 'time_step_%04d'
MODEL_TIME_STEP_DIRNAME = 'time_step_{:0>4}'

MODEL_PARAMETER_DIM = 7
MODEL_PARAMETER_LOWER_BOUND = np.zeros(MODEL_PARAMETER_DIM)
MODEL_PARAMETER_UPPER_BOUND = np.ones(MODEL_PARAMETER_DIM) * np.inf

# MODEL_PARAMETERS_SET_DIRNAME = 'parameter_set_%05d'                    # %0Xd is substituted by the number of the run to X digits
MODEL_PARAMETERS_SET_DIRNAME = 'parameter_set_{:0>5}'
MODEL_PARAMETERS_FILENAME = 'parameters.txt'
MODEL_PARAMETERS_FORMAT_STRING = '%.18f'
MODEL_PARAMETERS_MAX_DIFF = 10**(-10) * MODEL_PARAMETER_DIM

MODEL_SPINUP_DIRNAME = 'spinup'
MODEL_DERIVATIVE_DIRNAME = 'derivative'
# MODEL_PARTIAL_DERIVATIVE_DIRNAME = 'partial_derivative_%01d_%01d' # partial_derivative, h_factor
MODEL_PARTIAL_DERIVATIVE_DIRNAME = 'partial_derivative_{}_{:+}' # partial_derivative, h_factor
# MODEL_RUN_DIRNAME = 'run_%02d'              # %0Xd is substituted by the number of the run to X digits
MODEL_RUN_DIRNAME = 'run_{:0>2}'              # %0Xd is substituted by the number of the run to X digits
MODEL_RUN_OPTIONS_FILENAME = 'run_options.txt'
# MODEL_F_FILENAME = 'F_%04d.npy'                     # %0Xd is substituted by the time dimension to X digits
# MODEL_DF_FILENAME = 'DF_%04d_%01d.npy'              # t_dim, accuracy_order

MODEL_SPINUP_MAX_YEARS = 25000
MODEL_DERIVATIVE_SPINUP_YEARS = 50

# MODEL_INTERPOLATOR_FILE = os.path.join(MODEL_OUTPUT_DIR, 'interpolator_%01d.ppy')
MODEL_INTERPOLATOR_FILE = os.path.join(MODEL_OUTPUT_DIR, 'interpolator_{}.ppy')
MODEL_INTERPOLATOR_AMOUNT_OF_WRAP_AROUND = 1/2880
MODEL_INTERPOLATOR_NUMBER_OF_LINEAR_INTERPOLATOR = 0 #2882
MODEL_INTERPOLATOR_TOTAL_OVERLAPPING_OF_LINEAR_INTERPOLATOR = 0 #2


# METOS_PATH = os.path.join(BASE_DIR, 'metos3d/v0.2')
METOS_PATH = os.path.join(BASE_DIR, 'metos3d/v_rz_2')
METOS_LAND_SEA_MASK_FILE_PETSC = os.path.join(BASE_DIR, 'metos3d/landSeaMask.petsc')
METOS_LAND_SEA_MASK_FILE_NPY = os.path.join(BASE_DIR, 'metos3d/landSeaMask.npy')

# METOS_TRAJECTORY_FILENAMES = ('sp0000-ts%04d-dop_output.petsc', 'sp0000-ts%04d-po4_output.petsc')
METOS_TRAJECTORY_FILENAMES = ('sp0000-ts{:0>4}-dop_output.petsc', 'sp0000-ts{:0>4}-po4_output.petsc')
METOS_TRACER_DIM = len(METOS_TRAJECTORY_FILENAMES)

METOS_X_RANGE = [0, 360]
METOS_Y_RANGE = [-90, +90]
METOS_Z = [0, 50, 120, 220, 360, 550, 790, 1080, 1420, 1810, 2250, 2740, 3280, 3870, 4510]
# METOS_Z = [50, 120, 220, 360, 550, 790, 1080, 1420, 1810, 2250, 2740, 3280, 3870, 4510, 5200]
METOS_Z_CENTER = (np.array(METOS_Z[1:] + [9000]) + np.array(METOS_Z)) / 2
METOS_DIM = (128, 64, len(METOS_Z))
METOS_X_DIM = METOS_DIM[0]
METOS_Y_DIM = METOS_DIM[1]
METOS_Z_DIM = METOS_DIM[2]
METOS_T_RANGE = [0, 1]

