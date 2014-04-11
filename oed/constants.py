import os.path
from ndop.constants import OED_DIR


## constants for sample standard deviation
T_DIM = 52
MEASUREMENTS_BOXES_DICT_FILE = os.path.join(OED_DIR, 'measurement_boxes_dict.ppy')


## constants for interpolation
AMOUNT_OF_WRAP_AROUND = 0.1
NUMBER_OF_LINEAR_INTERPOLATOR = 2
TOTAL_OVERLAPPING_OF_LINEAR_INTERPOLATOR = 0.2
MEASUREMENTS_BOXES_DEVIATIONS_INTERPOLATION_FILE = os.path.join(OED_DIR, 'measurement_boxes_deviation_'+str(T_DIM)+'_'+str(AMOUNT_OF_WRAP_AROUND)+'_'+str(NUMBER_OF_LINEAR_INTERPOLATOR)+'_'+str(TOTAL_OVERLAPPING_OF_LINEAR_INTERPOLATOR)+'.npy')