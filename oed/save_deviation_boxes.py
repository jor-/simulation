import os.path
import logging
from ndop.oed.io import save_deviation_boxes as save
from ndop.oed.constants import OED_DIR

time_dim = 4
file = os.path.join(OED_DIR, 'measurement_boxes_deviation_' +  str(time_dim) + '_.npy')

logging.basicConfig(level=logging.DEBUG)
save(minimum_measurements=5, time_dim=time_dim, deviation_file=file)