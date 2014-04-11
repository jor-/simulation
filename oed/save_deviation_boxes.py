import os.path
import util.logging

from ndop.oed.io import save_deviation_boxes as save
from ndop.oed.constants import MEASUREMENTS_BOXES_DEVIATIONS_INTERPOLATION_FILE


file_prefix = os.path.splitext(MEASUREMENTS_BOXES_DEVIATIONS_INTERPOLATION_FILE)[0]
logging_file = file_prefix + '.log'
deviations_file = file_prefix + '.npy'

with util.logging.Logger(logging_file=logging_file):
    save(deviations_box_file=deviations_file)
