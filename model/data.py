import os
import itertools
import numpy as np
import logging

import measurements.land_sea_mask.load
# import measurements.util.regrid
# import util.pattern
import util.petsc.universal


## get land sea mask

def _check_land_sea_mask(land_sea_mask):
    from ndop.model.constants import METOS_DIM
    
    ## check input
    if land_sea_mask.ndim != 2:
        raise ValueError('The land sea mask must have 2 dimensions, but its shape is {}.'.format(land_sea_mask.shape))
    metos_shape = tuple(METOS_DIM[0:2])
    if land_sea_mask.shape != metos_shape:
        raise ValueError('The land sea mask must have the shape {}, but its shape is {}.'.format(metos_shape, land_sea_mask.shape))
#     assert land_sea_mask.ndim == 2
#     assert land_sea_mask.shape == METOS_DIM[0:2]


def load_land_sea_mask():
    return measurements.land_sea_mask.load.resolution_128x64x15()
#     from ndop.model.constants import METOS_LAND_SEA_MASK_FILE_PETSC, METOS_LAND_SEA_MASK_FILE_NPY
#     
#     try:
#         land_sea_mask = np.load(METOS_LAND_SEA_MASK_FILE_NPY)
#         
#         logging.debug('Returning land-sea-mask loaded from {} file.'.format(METOS_LAND_SEA_MASK_FILE_NPY))
#         
#     except (OSError, IOError):
#         land_sea_mask = util.petsc.universal.load_petsc_mat_to_array(METOS_LAND_SEA_MASK_FILE_PETSC, dtype=int)
#         land_sea_mask = land_sea_mask.transpose() # metos3d: x and y are changed
#         
#         logging.debug('Saving land-sea-mask to {} file.'.format(METOS_LAND_SEA_MASK_FILE_NPY))
#         
#         np.save(METOS_LAND_SEA_MASK_FILE_NPY, land_sea_mask)
#         
#         logging.debug('Returning land-sea-mask loaded from petsc file.')
#     
#     _check_land_sea_mask(land_sea_mask)
#     
#     return land_sea_mask



## convert Metos vector to 3D vector

def convert_metos_1D_to_3D(metos_vec, land_sea_mask):
    from ndop.model.constants import METOS_Z_DIM
    
    x_dim, y_dim = land_sea_mask.shape
    
    ## init array
    array = np.empty([x_dim, y_dim, METOS_Z_DIM], dtype=np.float64)
    array.fill(np.nan)
    
    ## fill array
    logging.debug('Converting metos {} vector to {} matrix.'.format(metos_vec.shape, array.shape))
    
    offset = 0
    for iy in range(y_dim):
        for ix in range(x_dim):
            length = land_sea_mask[ix, iy]
            if not length == 0:
                array[ix, iy, 0:length] = metos_vec[offset:offset+length]
                offset = offset + length
    
    return array





## load trajectory

def load_trajectories_to_universal(path, convert_function=None, converted_result_shape=None, tracer_indices=None, time_dim_desired=None):
    from ndop.model.constants import MODEL_TIME_STEP_SIZE_MAX, METOS_TRAJECTORY_FILENAMES
    
    logging.debug('Loading trajectories with tracer indices {}, desired time dim {} and convert function {} with result shape {} from {}.'.format(tracer_indices,  time_dim_desired, convert_function, converted_result_shape, path))
    
    
    ## check input
    
#     # check time_step_size
#     if MODEL_TIME_STEP_SIZE_MAX % time_step_size != 0:
#         raise ValueError('The time step size {} must be wrong. It has to be a divider of {}.'.format(time_step_size, MODEL_TIME_STEP_SIZE_MAX))
#     
#     assert MODEL_TIME_STEP_SIZE_MAX % time_step_size == 0
#     
#     
#     # check t_dim
#     number_of_petsc_vecs = MODEL_TIME_STEP_SIZE_MAX / time_step_size
    
    
    # check tracer_indices
    tracer_all_len = len(METOS_TRAJECTORY_FILENAMES)
    
    if tracer_indices is not None:
        tracer_indices = np.asanyarray(tracer_indices, dtype=np.int)
        if tracer_indices.ndim == 0:
            tracer_indices = tracer_indices.reshape(1)
        
        tracer_indices_min = np.min(tracer_indices)
        tracer_indices_min_allowed = 0
        if tracer_indices_min < tracer_indices_min_allowed:
            raise ValueError('The tracer indices {} are not allowed. Each index must be greater or equal to {}.'.format(tracer_indices, tracer_indices_min_allowed))
        tracer_indices_max = np.max(tracer_indices)
        tracer_indices_max_allowed = tracer_all_len - 1
        if tracer_indices_max > tracer_indices_max_allowed:
            raise ValueError('The tracer indices {} are not allowed. Each index must be less or equal to {}.'.format(tracer_indices, tracer_indices_max_allowed))
    else:
        tracer_indices = np.arange(tracer_all_len)
    
    assert np.all(tracer_indices >= 0) and np.all(tracer_indices <= tracer_all_len - 1)
    
    # check convert_function
    if convert_function is None:
        convert_function = lambda x: x
        if converted_result_shape is not None:
            raise ValueError('The convert function is None but the converted result shape is not None ({}).'.format(converted_result_shape))
    elif not callable(convert_function):
        raise ValueError('The convert function {} has to be callable.'.format(convert_function))
    
    assert callable(convert_function)
    
    
    ## calculate number_of_petsc_vecs
    number_of_petsc_vecs = MODEL_TIME_STEP_SIZE_MAX
    number_of_petsc_vecs_found = False
    while not number_of_petsc_vecs_found:
#         filename = util.pattern.replace_int_pattern(METOS_TRAJECTORY_FILENAMES[0], number_of_petsc_vecs - 1)
        filename = METOS_TRAJECTORY_FILENAMES[0].format(number_of_petsc_vecs - 1)
        file = os.path.join(path, filename)
        if not os.path.exists(file):
            if number_of_petsc_vecs > 1:
                number_of_petsc_vecs -= 1
            else:
                raise FileNotFoundError('No PETSc vectors found in {}.'.format(path))
        else:
            number_of_petsc_vecs_found = True
    
    logging.debug('{} petsc vectors were found for each tracer.'.format(number_of_petsc_vecs_found))
    
    ## calculate t_step, check time_dim_desired
    if time_dim_desired is not None:
        if number_of_petsc_vecs % time_dim_desired == 0:
            t_step = int(number_of_petsc_vecs / time_dim_desired)
        else:
            raise ValueError('The desired time dimension {0} can not be satisfied because {1} is not divisible by {0}.'.format(time_dim_desired, number_of_petsc_vecs))
    else:
        time_dim_desired = number_of_petsc_vecs
        t_step = 1
    
    assert number_of_petsc_vecs % time_dim_desired == 0
    
    
    
    
    
    ## init trajectory
    if converted_result_shape is None:
        filename = METOS_TRAJECTORY_FILENAMES[0].format(0)
        file = os.path.join(path, filename)
        trajectory = util.petsc.universal.load_petsc_vec_to_array(file)
        converted_result_shape = convert_function(trajectory).shape
    
    tracer_indices_len = len(tracer_indices)
    trajectory_shape = (tracer_indices_len, time_dim_desired) + converted_result_shape
    trajectory = np.zeros(trajectory_shape, dtype=np.float64)
    
    
    ## load and calculate trajectory
    logging.debug('Loading trajectories from {} to array of size {}.'.format(path, trajectory.shape))
    
    for tracer_indices_index in range(tracer_indices_len):
        tracer_index = tracer_indices[tracer_indices_index]
        
        logging.debug('Loading trajectory for tracer {}.'.format(tracer_index))
        file_pattern = METOS_TRAJECTORY_FILENAMES[tracer_index]
        for time_index in range(time_dim_desired):
#             logging.debug('Loading trajectory for time {}.'.format(time_index))
            
            ## average trajectory
            for k in range(t_step):
                file_nr = time_index * t_step + k
                filename = file_pattern.format(file_nr)
                file = os.path.join(path, filename)
                if k == 0:
                    trajectory_averaged = util.petsc.universal.load_petsc_vec_to_array(file)
                else:
                    trajectory_averaged += util.petsc.universal.load_petsc_vec_to_array(file)
                
            trajectory_averaged /= t_step
            
            ## convert trajectory
            trajectory_averaged = convert_function(trajectory_averaged)
            assert trajectory_averaged.shape == converted_result_shape
            
            trajectory[tracer_indices_index, time_index] = trajectory_averaged
    
    logging.debug('Trajectory with shape {} loaded.'.format(trajectory.shape))
    
    return trajectory


def _check_tracer_index(tracer_index):
    from ndop.model.constants import METOS_TRACER_DIM
    
    tracer_index_array = np.asanyarray(tracer_index, dtype=np.int)
    if tracer_index_array.ndim != 0:
        raise ValueError('The tracer index must be an int, but its {}.'.format(tracer_index))
    if tracer_index < 0 or tracer_index >= METOS_TRACER_DIM:
        raise ValueError('The tracer index must be between 0 and {}, but its {}.'.format(METOS_TRACER_DIM-1, tracer_index))
        
    


def load_trajectories_to_index_array(path, tracer_index, land_sea_mask, time_dim_desired=None):
    from ndop.model.constants import METOS_DIM
    
#     
#     ## check input
#     if land_sea_mask.ndim != 2:
#         raise ValueError('The land sea mask must have 2 dimensions but its shape is {}.'.format(land_sea_mask.shape))
#     assert land_sea_mask.ndim == 2
#     assert land_sea_mask.shape == METOS_DIM[0:2]
#     
#     tracer_index_array = np.asanyarray(tracer_index, dtype=np.int)
#     if tracer_index_array.ndim != 0:
#         raise ValueError('The tracer index must be an int, but its {}.'.format(tracer_index))
    
    
    ## check input
    _check_land_sea_mask(land_sea_mask)
    _check_tracer_index(tracer_index)
    
    ## load trajectory
    convert_function = lambda metos_vec: convert_metos_1D_to_3D(metos_vec, land_sea_mask)
    trajectory = load_trajectories_to_universal(path, convert_function=convert_function, converted_result_shape=METOS_DIM, tracer_indices=tracer_index, time_dim_desired=time_dim_desired)
    trajectory = trajectory[0]
    
    assert trajectory.ndim == 4
    
    return trajectory



def load_trajectories_to_point_array(path, tracer_index, land_sea_mask, time_dim_desired=None):
    from ndop.model.constants import METOS_DIM, METOS_Z_CENTER
    
    ## check input
    _check_land_sea_mask(land_sea_mask)
    _check_tracer_index(tracer_index)
    
    ## load trajectory with universal function
    def convert_function(metos_vec):
        trajectory = convert_metos_1D_to_3D(metos_vec, land_sea_mask)
        data_mask = np.logical_not(np.isnan(trajectory))
        
        data_indices = np.array(np.where(data_mask)).swapaxes(0, 1)
        assert data_indices.ndim == 2
        assert data_indices.shape[1] == 3
        
        data_values = trajectory[data_mask].reshape([len(data_indices), 1])
        assert data_values.ndim == 2
        assert data_values.shape[1] == 1
        assert len(data_indices) == len(data_values)
        
        data = np.concatenate([data_indices, data_values], axis=1)
        assert data.ndim == 2
        assert data.shape[1] == 4
        
        return data
    
    trajectory = load_trajectories_to_universal(path, convert_function=convert_function, tracer_indices=tracer_index, time_dim_desired=time_dim_desired)
    trajectory = trajectory[0]
    assert trajectory.ndim == 3
    assert np.all(trajectory[:, :, :3] % 1 == 0)
    
    ## convert time index to point value
    t_dim, point_len_per_t, point_dim  = trajectory.shape
    trajectory_point_array = np.empty((t_dim * point_len_per_t, point_dim + 1))
    for t_index in range(t_dim):
        trajectory_point_array[t_index*point_len_per_t : (t_index+1)*point_len_per_t, 0] = t_index / t_dim
        trajectory_point_array[t_index*point_len_per_t : (t_index+1)*point_len_per_t, 1:] = trajectory[t_index]
    
    ## convert metos indices to point values
    trajectory_point_array[:, 1] *= 360 / METOS_DIM[1]
    trajectory_point_array[:, 2] *= 180 / METOS_DIM[2]
    trajectory_point_array[:, 3] = METOS_Z_CENTER[trajectory_point_array[:, 3].astype(dtype=np.int)]
    
    assert trajectory_point_array.ndim == 2
    assert trajectory_point_array.shape[1] == 5
    
    return trajectory_point_array








# def convert_point_to_box_index(t, x, y, z, t_dim, land_sea_mask):
#     from ndop.model.constants import METOS_T_RANGE, METOS_X_RANGE, METOS_Y_RANGE, METOS_Z
#     
#     return measurements.util.regrid.convert_point_to_box_index(t, x, y, z, t_dim, land_sea_mask, METOS_Z, t_range=METOS_T_RANGE, x_range=METOS_X_RANGE, y_range=METOS_Y_RANGE)