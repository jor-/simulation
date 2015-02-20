import os.path
import util.logging

from ndop.oed.deviation.io import save_po4_interpolated_deviation_list as save
from ndop.oed.deviation.constants import PO4_MEASUREMENTS_BOXES_INTERPOLATED_DEVIATION_LIST_FILE


file_prefix = os.path.splitext(PO4_MEASUREMENTS_BOXES_INTERPOLATED_DEVIATION_LIST_FILE)[0]
log_file = file_prefix + '.log'

with util.logging.Logger(log_file=log_file, disp_stdout=False):
    save(interpolated_deviation_list_file=PO4_MEASUREMENTS_BOXES_INTERPOLATED_DEVIATION_LIST_FILE)
