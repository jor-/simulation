import os.path
import util.logging

from ndop.oed.deviation.io import save_po4_measurement_boxes_dict as save
from ndop.oed.deviation.constants import PO4_MEASUREMENTS_BOXES_DICT_FILE


file_prefix = os.path.splitext(PO4_MEASUREMENTS_BOXES_DICT_FILE)[0]
log_file = file_prefix + '.log'
deviations_file = file_prefix + '.npy'

with util.logging.Logger(log_file=log_file, disp_stdout=False):
    save()