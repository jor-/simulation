import numpy as np

JOB_OPTIONS_FILENAME = 'job_options.hdf5'
JOB_MEMORY_GB = 4


MODEL_OUTPUTS_PATH = '/work_j2/sunip229/NDOP/model_outputs'
MODEL_PARAMETERS_SET_DIRECTORY = 'parameter_set_%05d'                    # %0Xd is substituted by the number of the run to X digits
MODEL_PARAMETERS_FILENAME = 'parameters.txt'
MODEL_PARAMETERS_FORMAT_STRING = '%.18f'
MODEL_TIME_STEP_SIZE_MAX = 2880

MODEL_SPINUP_DIRECTORY = 'spinup'
MODEL_DERIVATIVE_DIRECTORY = 'derivative'
MODEL_PARTIAL_DERIVATIVE_DIRECTORY = 'partial_derivative_%01d_%01d' # partial_derivative, h_factor_index
MODEL_RUN_DIRECTORY = 'run_%02d'              # %0Xd is substituted by the number of the run to X digits
MODEL_RUN_OPTIONS_FILENAME = 'run_options.txt'
# MODEL_SPINUP_YEARS_FILENAME = 'years.txt'
# MODEL_SPINUP_TOLERANCE_FILENAME = 'tolerance.txt'
# MODEL_SPINUP_OPTIONS_FILENAME = 'python_options.hdf5'
MODEL_F_FILENAME = 'F_%04d.npy'                      # %0Xd is substituted by the time dimension to X digits
MODEL_DF_FILENAME = 'DF_%04d_%01d.npy'              # t_dim, accuracy_order

MODEL_PARAMETER_DIM = 7
MODEL_PARAMETER_LOWER_BOUND = np.zeros(MODEL_PARAMETER_DIM)
MODEL_PARAMETER_UPPER_BOUND = np.ones(MODEL_PARAMETER_DIM) * np.inf
MODEL_DERIVATIVE_SPINUP_YEARS = 100



METOS_LAND_SEA_MASK_FILE_PETSC = '/work_j2/sunip229/NDOP/metos3d_data/landSeaMask.petsc'
METOS_LAND_SEA_MASK_FILE_NPY = '/work_j2/sunip229/NDOP/metos3d_data/landSeaMask.npy'

METOS_TRAJECTORY_FILENAMES = ('sp0000-ts%04d-dop_output.petsc', 'sp0000-ts%04d-po4_output.petsc')

METOS_X_RANGE = [0, 360]
# METOS_X_DIM = 124
METOS_Y_RANGE = [-90, +90]
# METOS_X_DIM = 64
METOS_Z = [0, 50, 120, 220, 360, 550, 790, 1080, 1420, 1810, 2250, 2740, 3280, 3870, 4510]
# METOS_Z = [50, 120, 220, 360, 550, 790, 1080, 1420, 1810, 2250, 2740, 3280, 3870, 4510, 5200]
METOS_Z_DIM = len(METOS_Z)
METOS_T_RANGE = [0, 1]

