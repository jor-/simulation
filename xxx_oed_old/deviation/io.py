import os.path
import numpy as np

import logging
logger = logging.getLogger(__name__)

import measurements.dop.pw.deviation
import measurements.po4.wod.data.io
from measurements.po4.wod.data.results import Measurements_Unsorted as Measurements
from measurements.po4.wod.deviation.model import Deviation_Model
import measurements.util.map

from ndop.model.constants import METOS_X_DIM, METOS_Y_DIM, METOS_Z_LEFT
from .constants import PO4_MEASUREMENTS_BOXES_DICT_FILE, T_DIM, PO4_MEASUREMENTS_BOXES_INTERPOLATED_DEVIATION_LIST_FILE, PO4_MEASUREMENTS_BOXES_INTERPOLATED_DEVIATION_MAP_FILE



def save_po4_measurement_boxes_dict(measurement_box_file=PO4_MEASUREMENTS_BOXES_DICT_FILE):
    m = measurements.po4.wod.data.io.load_measurement_dict_unsorted()
    m.categorize_indices((1./2880,))
    m.transform_indices_to_boxes(METOS_X_DIM, METOS_Y_DIM, METOS_Z_LEFT)
    m.save(measurement_box_file)
    logger.debug('Measurement box dict saved at {}.'.format(measurement_box_file))


def load_po4_measurement_boxes_dict(measurement_box_file=PO4_MEASUREMENTS_BOXES_DICT_FILE):
    m = Measurements()
    m.load(measurement_box_file)
    logger.debug('Measurement box dict loaded from {}'.format(measurement_box_file))
    return m



def save_po4_interpolated_deviation_list(t_dim=T_DIM, measurements_box_file=PO4_MEASUREMENTS_BOXES_DICT_FILE, interpolated_deviation_list_file=PO4_MEASUREMENTS_BOXES_INTERPOLATED_DEVIATION_LIST_FILE):
    
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
    
    np.save(interpolated_deviation_list_file, result)
    logger.debug('Standard deviation in list shape saved to {}.'.format(interpolated_deviation_list_file))


def load_po4_interpolated_deviation_list(interpolated_deviation_list_file=PO4_MEASUREMENTS_BOXES_INTERPOLATED_DEVIATION_LIST_FILE):
    deviation = np.load(interpolated_deviation_list_file)
    logger.debug('Standard deviation in list shape loaded from {}.'.format(interpolated_deviation_list_file))
    return deviation



def save_interpolated_deviation_map(interpolated_deviation_list_file=PO4_MEASUREMENTS_BOXES_INTERPOLATED_DEVIATION_LIST_FILE, interpolated_deviation_map_file=PO4_MEASUREMENTS_BOXES_INTERPOLATED_DEVIATION_MAP_FILE):
    deviation_list = load_po4_interpolated_deviation_list(interpolated_deviation_list_file=interpolated_deviation_list_file)
    
    m = Measurements()
    m.add_results(deviation_list[:,:-1], deviation_list[:,-1])
    deviation_map = measurements.util.map.insert_values_in_map(m.means(), no_data_value=np.inf)
    
    np.save(interpolated_deviation_map_file, deviation_map)
    logger.debug('Standard deviation in map shape saved to {}.'.format(interpolated_deviation_map_file))


#TODO load from disk
# def load_interpolated_deviation_map(t_dim=T_DIM, interpolated_deviation_map_file=PO4_MEASUREMENTS_BOXES_INTERPOLATED_DEVIATION_MAP_FILE):
#     deviation_map = np.load(interpolated_deviation_map_file)
#     logger.debug('Standard deviation in map shape loaded from {}.'.format(interpolated_deviation_map_file))
#     
#     return deviation_map


def load_interpolated_deviation_map(t_dim=T_DIM):
    ## PO4
    po4_deviation_list = load_po4_interpolated_deviation_list(interpolated_deviation_list_file=PO4_MEASUREMENTS_BOXES_INTERPOLATED_DEVIATION_LIST_FILE)
    m = Measurements()
    m.add_results(po4_deviation_list[:,:-1], po4_deviation_list[:,-1])
    m.categorize_indices((1./t_dim,))
    po4_deviation_map = measurements.util.map.insert_values_in_map(m.means(), no_data_value=np.inf)
    
    ## DOP
    dop_deviation_map = np.ones_like(po4_deviation_map) * measurements.dop.pw.deviation.get_average_deviation()
    
    ### concatenate
    deviation_map = np.concatenate([dop_deviation_map[np.newaxis, :], po4_deviation_map[np.newaxis, :]], axis=0)
    
    return deviation_map