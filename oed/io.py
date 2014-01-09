import numpy as np

import measurements.po4.wod.data.io
from measurements.po4.wod.data.results import Measurements_Unsorted
from measurements.po4.wod.deviation.model import Deviation_Model
import measurements.util.map

from ndop.metos3d.constants import METOS_X_DIM, METOS_Y_DIM, METOS_Z
from constants import MEASUREMENTS_BOXES_DICT_FILE, MEASUREMENTS_BOXES_DEVIATIONS_FILE, MEASUREMENTS_BOXES_DEVIATIONS_MIN_MEASUREMENTS, MEASUREMENTS_BOXES_DEVIATIONS_TIME_DIM

import logging
logger = logging.getLogger(__name__)


def save_measurement_boxes_dict(measurement_file=MEASUREMENTS_BOXES_DICT_FILE):
    m = measurements.po4.wod.data.io.load_measurement_dict_unsorted()
    m.categorize_indices((1./2880,))
    m.transform_indices_to_boxes(METOS_X_DIM, METOS_Y_DIM, METOS_Z)
    m.save(measurement_file)
    logger.debug('Measurement box dict saved at %s' % measurement_file)


def load_measurement_boxes_dict(measurement_file=MEASUREMENTS_BOXES_DICT_FILE):
    m = Measurements_Unsorted()
    m.load(measurement_file)
    logger.debug('Measurement box dict loaded from %s' % measurement_file)
    return m



def save_deviation_boxes(minimum_measurements=MEASUREMENTS_BOXES_DEVIATIONS_MIN_MEASUREMENTS, time_dim=MEASUREMENTS_BOXES_DEVIATIONS_TIME_DIM, measurements_file=MEASUREMENTS_BOXES_DICT_FILE, deviation_file=MEASUREMENTS_BOXES_DEVIATIONS_FILE):
    
    logger.debug('Calculating standard deviation for ndop boxes with min %d measurements and time dim %d.' % (minimum_measurements, time_dim))
    
    ## prepare points
    masked_map = measurements.util.map.init_masked_map()
    sea_indices = np.array(np.where(np.logical_not(np.isnan(masked_map)))).transpose()
    sea_indices_len = sea_indices.shape[0]
    logger.debug('Found %d sea points.' % sea_indices_len)
    points = np.empty((sea_indices_len * time_dim, sea_indices.shape[1] + 1))
    for i in range(time_dim):
        points[i*sea_indices_len:(i+1)*sea_indices_len, 0] = i / float(time_dim)
        points[i*sea_indices_len:(i+1)*sea_indices_len, 1:] = sea_indices
    logger.debug('Interpolation points shape is %s.' % str(points.shape))
    
    
    ## calculate deviation
    deviation_model = Deviation_Model(minimum_measurements=minimum_measurements, separation_values=(1./time_dim, None, None, None), t_range=[0, 1], x_range=[0, METOS_X_DIM], measurements_file=measurements_file, convert_spherical_to_cartesian=False)
    
    deviation = deviation_model.deviation(points)
    deviation = deviation.reshape(len(deviation), 1)
    logger.debug('Standard deviation shape is %s.' % str(deviation.shape))
    
    logger.debug('Saving standard deviation at %d points to %s.' % (len(deviation), deviation_file))
    result = np.concatenate((points, deviation), axis=1)
    np.save(deviation_file, result)
    
    logger.debug('Standard deviation for ndop boxes saved.')
