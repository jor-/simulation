import os.path
from ndop.constants import OED_DIR

MEASUREMENTS_BOXES_DICT_FILE = os.path.join(OED_DIR, 'measurement_boxes_dict.ppy')
MEASUREMENTS_BOXES_DEVIATIONS_MIN_MEASUREMENTS = 5
MEASUREMENTS_BOXES_DEVIATIONS_TIME_DIM = 52
MEASUREMENTS_BOXES_DEVIATIONS_FILE = os.path.join(OED_DIR, 'measurement_boxes_deviation.npy')