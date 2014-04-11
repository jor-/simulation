import numpy as np
import os.path

import ndop.model.direct_access
from ndop.constants import MODEL_OUTPUT_DIR

import util.plot

import logging
logger = logging.getLogger(__name__)

def plot_model_output(path='/tmp', vmax=(None, None)):
    logger.debug('Plotting model output')
    parameter_set_dir = os.path.join(MODEL_OUTPUT_DIR, 'time_step_0001', 'parameter_set_00000')
    f = ndop.model.direct_access.f(parameter_set_dir)
    file = os.path.join(path, 'model_dop_output.png')
    util.plot.data(f[0], file, land_value=np.nan, no_data_value=None, vmin=0, vmax=vmax[0])
    file = os.path.join(path, 'model_po4_output.png')
    util.plot.data(f[1], file, land_value=np.nan, no_data_value=None, vmin=0, vmax=vmax[1])
