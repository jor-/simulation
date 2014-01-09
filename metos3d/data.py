import os
import math
import bisect
import itertools
import numpy as np

import logging
logger = logging.getLogger(__name__)

import util.pattern
import util.interpolation



def load_land_sea_mask():
    from ndop.metos3d.constants import METOS_LAND_SEA_MASK_FILE_PETSC, METOS_LAND_SEA_MASK_FILE_NPY
    
    try:
        land_sea_mask = np.load(METOS_LAND_SEA_MASK_FILE_NPY)
        
        logger.debug('Returning land-sea-mask loaded from {} file.'.format(METOS_LAND_SEA_MASK_FILE_NPY))
        
    except (OSError, IOError):
        import util.petsc
        
        land_sea_mask = util.petsc.load_petsc_mat_to_array(METOS_LAND_SEA_MASK_FILE_PETSC, dtype=int)
        land_sea_mask = land_sea_mask.transpose() # metos3d: x and y are changed
        
        logger.debug('Saving land-sea-mask to {} file.'.format(METOS_LAND_SEA_MASK_FILE_NPY))
        
        np.save(METOS_LAND_SEA_MASK_FILE_NPY, land_sea_mask)
        
        logger.debug('Returning land-sea-mask loaded from petsc file.')
    
    return land_sea_mask



def convert_1D_to_3D(metos_vec, land_sea_mask):
    from ndop.metos3d.constants import METOS_Z_DIM
    
    x_dim, y_dim = land_sea_mask.shape
    
    ## init array
    array = np.empty([x_dim, y_dim, METOS_Z_DIM], dtype=np.float64)
    array.fill(np.nan)
    
    ## fill array
    logger.debug('Converting metos {} vector to {} matrix.'.format(metos_vec.shape, array.shape))
    
    offset = 0
    for iy in range(y_dim):
        for ix in range(x_dim):
            length = land_sea_mask[ix, iy]
            if not length == 0:
                array[ix, iy, 0:length] = metos_vec[offset:offset+length]
                offset = offset + length
    
    return array
    


def load_trajectories(path, t_dim, time_step_size, land_sea_mask=None):
    import util.petsc
    from ndop.metos3d.constants import MODEL_TIME_STEP_SIZE_MAX, METOS_TRAJECTORY_FILENAMES
    
    number_of_petsc_vecs = MODEL_TIME_STEP_SIZE_MAX / time_step_size
    
    ## check t_dim
    if number_of_petsc_vecs % t_dim == 0:
        t_step = int(number_of_petsc_vecs / t_dim)
    else:
        raise ValueError('The desired time dimension {0} can not be satisfied because {1} is not divisible by {0}.'.format(t_dim, number_of_petsc_vecs))
    
    
    ## init trajectory
    tracer_dim = len(METOS_TRAJECTORY_FILENAMES)
    filename = util.pattern.replace_int_pattern(METOS_TRAJECTORY_FILENAMES[0], 0)
    file = os.path.join(path, filename)
    trajectory = util.petsc.load_petsc_vec_to_array(file)
    if land_sea_mask is not None:
        trajectory = convert_1D_to_3D(trajectory, land_sea_mask)
    s_dim = trajectory.shape
    trajectory_shape = (tracer_dim, t_dim) + s_dim
    trajectory = np.zeros(trajectory_shape, dtype=np.float64)
    
    
    ## load and calculate trajectory
    logger.debug('Loading trajectories from {} of size {}.'.format(path, trajectory.shape))
    
    for i in range(tracer_dim):
        logger.debug('Loading trajectory for tracer {}.'.format(i))
        file_pattern = METOS_TRAJECTORY_FILENAMES[i]
        for j in range(t_dim):
            logger.debug('Loading trajectory for time {}.'.format(j))
            for k in range(t_step):
                file_nr = j * t_step + k
                filename = util.pattern.replace_int_pattern(file_pattern, file_nr)
                file = os.path.join(path, filename)
                if k == 0:
                    trajectory_averaged = util.petsc.load_petsc_vec_to_array(file)
                else:
                    trajectory_averaged += util.petsc.load_petsc_vec_to_array(file)
                
            trajectory_averaged /= t_step
            
            if land_sea_mask is not None:
                trajectory_averaged = convert_1D_to_3D(trajectory_averaged, land_sea_mask)
            
            trajectory[i, j] = trajectory_averaged
        
    return trajectory





def get_spatial_float_index(x, y, z, land_sea_mask):
    from ndop.metos3d.constants import METOS_X_RANGE, METOS_Y_RANGE, METOS_Z
    
    logger.debug('Getting spatial float index for {}.'.format((x, y, z)))
    
    ## adjust x coordinates if negative
    if x < 0:
        x += 360
    
    ## check input
    if x < METOS_X_RANGE[0] or x > METOS_X_RANGE[1]:
        raise ValueError('Value {} of x is not in range {}.'.format(x, METOS_X_RANGE))
    if y < METOS_Y_RANGE[0] or y > METOS_Y_RANGE[1]:
        raise ValueError('Value {} of y is not in range {}.'.format(y, METOS_Y_RANGE))
    if z < METOS_Z[0]:
        raise ValueError('Value {} of z have to be greater or equal to {}.'.format(z, METOS_Z_RANGE[0]))
    
    ## linear interpolate x and y index
    (x_dim, y_dim) = land_sea_mask.shape
    x_index_float = util.interpolation.get_float_index_for_equidistant_value(x, METOS_X_RANGE, x_dim)
    y_index_float = util.interpolation.get_float_index_for_equidistant_value(y, METOS_Y_RANGE, y_dim)
    
    ## lockup z
    z_index = bisect.bisect(METOS_Z, z) - 1
    
    if z_index + 1 < len(METOS_Z):
        z_left = METOS_Z[z_index]
        z_right = METOS_Z[z_index + 1]
        z_index_float = z_index + (z - z_left) / (z_right - z_left)
    else:
        z_index_float = z_index
    
    
    logger.debug('Float indices for {} are {}.'.format((x, y, z), (x_index_float, y_index_float, z_index_float)))
    
    return (x_index_float, y_index_float, z_index_float)



def get_temporal_float_index(t, t_dim):
    from ndop.metos3d.constants import METOS_T_RANGE
    
    logger.debug('Getting temporal float index for {}.'.format(t))
    
    ## check input
    if t < METOS_T_RANGE[0] or t > METOS_T_RANGE[1]:
        raise ValueError('Value {} of t is not in range {}.'.format(t, METOS_T_RANGE))
    
    ## interpolate
    t_index_float = util.interpolation.get_float_index_for_equidistant_value(t, METOS_T_RANGE, t_dim)
    
    logger.debug('Temporal float index for {} is {}.'.format(t, t_index_float))
    
    return t_index_float



def is_water(spatial_indices, land_sea_mask):
    logger.debug('Checking if indices are water.')
    
    ## check input
    spatial_indices = np.array(spatial_indices)
    
    dim = len(spatial_indices.shape)
    
    if dim > 2 or dim == 0:
        raise ValueError('The spatial_indices array has to have 1 or 2 dimensions, but it has {} dimensions.'.format(dim))
    elif dim == 1:
        spatial_len = spatial_indices.shape[0]
        if spatial_len != 3:
            raise ValueError('The len of spatial_indices array has to be 3, but it is {}.'.format(spatial_len))
        spatial_indices = spatial_indices.reshape(1, spatial_len)
    else:
        spatial_len = spatial_indices.shape[1]
        if spatial_len != 3:
            raise ValueError('The second dimension of the spatial_indices array to be 3, but it is {}.'.format(spatial_len))
    
    ## compute if water
    land_sea_mask_indices = spatial_indices[:,:spatial_len-1]
    land_sea_mask_values = land_sea_mask[land_sea_mask_indices[:,0], land_sea_mask_indices[:,1]]
    is_water = np.logical_or(np.isnan(land_sea_mask_values),  land_sea_mask_values >= spatial_indices[:,spatial_len-1])
    
    return is_water



def get_adjacent_water_indices(t, x, y, z, t_dim, land_sea_mask):
    logger.debug('Getting adjacent water indices for value {}.'.format((t, x, y, z)))
    
    spatial_float_indices = get_spatial_float_index(x, y, z, land_sea_mask)
    
    ## compute floored and ceiled spatial indices
    spatial_indices = ()
    
    for index in spatial_float_indices:
        index_floored = math.floor(index)
        index_ceiled = math.ceil(index)
        
        if index_floored != index_ceiled:
            spatial_indices += ((index_floored, index_ceiled),)
        else:
            spatial_indices += ((index_floored,),)
    
    ## combine spatial indices with cartesian product
    spatial_indices_combined = np.array(tuple(itertools.product(*spatial_indices)))
    
    ## get only water indices
    spatial_water_indices = spatial_indices_combined[is_water(spatial_indices_combined, land_sea_mask)]
    number_of_spatial_water_indices = spatial_water_indices.shape[0]
    
    logger.debug('Found {} spatial water indices.'.format(number_of_spatial_water_indices))
    
    
    ## combine water indices with temporal indices
    t_index_float =  get_temporal_float_index(t, t_dim)
    t_index_floored = math.floor(t_index_float)
    t_index_ceiled = math.ceil(t_index_float)
    if t_index_floored != t_index_ceiled:
        t_indices = (t_index_floored, t_index_ceiled)
    else:
        t_indices = (t_index_floored, )
    
    water_indices = np.empty((0, 4))
    
    for t_index in t_indices:
        t_index_array = np.ones((number_of_spatial_water_indices, 1)) * t_index
        water_indices = np.concatenate((water_indices, np.concatenate((t_index_array, spatial_water_indices), axis=1)))
    
    logger.debug('Returning {} water indices.'.format(water_indices.shape[0]))
    
    return water_indices





def get_all_water_boxes(land_sea_mask):
    logger.debug('Getting all water boxes.')
    
    (water_x, water_y) = np.where(land_sea_mask != 0)
    water_len = np.sum(land_sea_mask)
    water_boxes = np.empty([water_len, 3], dtype=np.int)
    
    j = 0
    for i in range(len(water_x)):
        x_i = water_x[i]
        y_i = water_y[i]
        z_i = land_sea_mask[x_i, y_i]
        
        j_range = range(j, j + z_i)
        water_boxes[j_range, 0] = x_i
        water_boxes[j_range, 1] = y_i
        water_boxes[j_range, 2] = range(z_i)
        
        j += z_i
    
    return water_boxes



def get_nearest_water_box(land_sea_mask, x_index, y_index, z_index):
    logger.debug('Getting nearest water box for index {}.'.format((x_index, y_index, z_index)))
    
    water_boxes = get_all_water_boxes(land_sea_mask)
    
    index = [x_index, y_index, z_index]
    nearest_water_box = util.interpolation.get_nearest_value_in_array(water_boxes, index)
    
    return nearest_water_box



def get_nearest_spatial_water_index(x, y, z, land_sea_mask):
    (x_index_float, y_index_float, z_index_float) = get_spatial_float_index(x, y, z, land_sea_mask)
    
    ## floor indices to int
    x_index = math.floor(x_index_float)
    y_index = math.floor(y_index_float)
    z_index = math.floor(z_index_float)
    
    ## get nearest water box if box is land
#     box_value = land_sea_mask[x_index, y_index]
#     if box_value is np.nan or box_value < z_index:
    if not is_water((x_index, y_index, z_index), land_sea_mask):
        logger.debug('Box {} is land.'.format((x_index, y_index, z_index)))
        (x_index, y_index, z_index) = get_nearest_water_box(land_sea_mask, x_index_float, y_index_float, z_index_float)
    
    logger.debug('Nearest index for {} is {}.'.format((x, y, z), (x_index_float, y_index_float, z_index_float)))
    
    return (x_index, y_index, z_index)



def get_nearest_temporal_index(t, t_dim):
    t_index_float = get_temporal_float_index(t, t_dim)
    t_index = math.floor(t_index_float)
    
    logger.debug('Nearest temporal index for {} is {}.'.format(t, t_index))
    
    return t_index



def get_nearest_water_index(t, x, y, z, t_dim, land_sea_mask):
    t_index =  get_nearest_temporal_index(t, t_dim)
    x_index, y_index, z_index = get_nearest_spatial_water_index(x, y, z, land_sea_mask)
    
    return (t_index, x_index, y_index, z_index)
