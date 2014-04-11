import os.path
import numpy as np

import logging
logger = logging.getLogger(__name__)

import measurements.po4.wod.data.io
from measurements.po4.wod.data.results import Measurements_Unsorted
from measurements.po4.wod.deviation.model import Deviation_Model
import measurements.util.map

from ndop.model.constants import METOS_X_DIM, METOS_Y_DIM, METOS_Z
from .constants import MEASUREMENTS_BOXES_DICT_FILE, MEASUREMENTS_BOXES_DEVIATIONS_INTERPOLATION_FILE, T_DIM



def save_measurement_boxes_dict(measurement_box_file=MEASUREMENTS_BOXES_DICT_FILE):
    m = measurements.po4.wod.data.io.load_measurement_dict_unsorted()
    m.categorize_indices((1./2880,))
    m.transform_indices_to_boxes(METOS_X_DIM, METOS_Y_DIM, METOS_Z)
    m.save(measurement_box_file)
    logger.debug('Measurement box dict saved at {].'.format(measurement_box_file))


def load_measurement_boxes_dict(measurement_box_file=MEASUREMENTS_BOXES_DICT_FILE):
    m = Measurements_Unsorted()
    m.load(measurement_box_file)
    logger.debug('Measurement box dict loaded from {}'.format(measurement_box_file))
    return m



def save_deviation_boxes(t_dim=T_DIM, measurements_box_file=MEASUREMENTS_BOXES_DICT_FILE, deviations_box_file=MEASUREMENTS_BOXES_DEVIATIONS_INTERPOLATION_FILE):
    
    from .constants import NUMBER_OF_LINEAR_INTERPOLATOR, TOTAL_OVERLAPPING_OF_LINEAR_INTERPOLATOR, AMOUNT_OF_WRAP_AROUND
    
    logger.debug('Calculating standard deviation for ndop boxes with time dim {}.'.format(t_dim))
    
    
    ## prepare points
    masked_map = measurements.util.map.init_masked_map()
    sea_indices = np.array(np.where(np.logical_not(np.isnan(masked_map)))).transpose()
    sea_indices_len = sea_indices.shape[0]
    logger.debug('Found {} sea points.'.format(sea_indices_len))
    points = np.empty((t_dim * sea_indices_len, sea_indices.shape[1] + 1))
    for t in range(t_dim):
        points[t*sea_indices_len : (t+1)*sea_indices_len, 0] = float(t) / t_dim
        points[t*sea_indices_len : (t+1)*sea_indices_len, 1:] = sea_indices
    
    
    ## calculate deviation
    deviation_model = Deviation_Model(measurements_file=measurements_box_file, separation_values=(1./t_dim, None, None, None), t_len=1, x_len=METOS_X_DIM, convert_spherical_to_cartesian=False, wrap_around_amount=AMOUNT_OF_WRAP_AROUND, number_of_linear_interpolators=NUMBER_OF_LINEAR_INTERPOLATOR, total_overlapping_linear_interpolators=TOTAL_OVERLAPPING_OF_LINEAR_INTERPOLATOR)
    
    deviation = deviation_model.deviation(points)
    deviation = deviation.reshape([len(deviation), 1])
    result = np.concatenate((points, deviation), axis=1)
    np.save(deviations_box_file, result)
    
    logger.debug('Standard deviation for ndop boxes saved at {}.'.format(deviations_box_file))


def load_deviation_boxes(deviations_box_file=MEASUREMENTS_BOXES_DEVIATIONS_INTERPOLATION_FILE):
    deviation = np.load(deviations_box_file)
    logger.debug('Standard deviation for ndop boxes loaded from {}.'.format(deviations_box_file))
    return deviation