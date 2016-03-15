import os
import itertools
import numpy as np
import logging

import util.petsc.universal


## convert Metos vector to 3D vector

def convert_metos_1D_to_3D(metos_vec):
#     from ndop.model.constants import METOS_X_DIM, METOS_Y_DIM, METOS_Z_DIM
    from ndop.model.constants import LSM

    ## init array
    array = np.empty([LSM.x_dim, LSM.y_dim, LSM.z_dim], dtype=np.float64)
    array.fill(np.nan)

    ## fill array
    logging.debug('Converting metos {} vector to {} matrix.'.format(metos_vec.shape, array.shape))

    offset = 0
    for iy in range(LSM.y_dim):
        for ix in range(LSM.x_dim):
            length = LSM[ix, iy]
            if length > 0:
                array[ix, iy, 0:length] = metos_vec[offset:offset+length]
                offset = offset + length

    return array



def convert_3D_to_metos_1D(data):
    assert data.ndim == 3

    metos_vec_len = np.sum(~np.isnan(data))
    metos_vec = np.empty(metos_vec_len)
    x_dim, y_dim, z_dim = data.shape

    offset = 0
    for iy in range(y_dim):
        for ix in range(x_dim):
            data_x_y = data[ix, iy]
            mask = ~ np.isnan(data_x_y)
            length = sum(mask)
            if length > 0:
                metos_vec[offset:offset+length] = data_x_y[mask]
                offset = offset + length

    return metos_vec



## load trajectory

def load_trajectories_to_universal(path, convert_function=None, converted_result_shape=None, tracer_indices=None, time_dim_desired=None, set_negative_values_to_zero=False):
    from ndop.model.constants import METOS_T_DIM, METOS_TRAJECTORY_FILENAMES

    logging.debug('Loading trajectories with tracer indices {}, desired time dim {}, set_negative_values_to_zero {} and convert function {} with result shape {} from {}.'.format(tracer_indices, time_dim_desired, set_negative_values_to_zero, convert_function, converted_result_shape, path))


    ## check input

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


    ## calculate tracer_time_dim
    tracer_time_dim = METOS_T_DIM
    tracer_time_dim_found = False
    while not tracer_time_dim_found:
        filename = METOS_TRAJECTORY_FILENAMES[0].format(tracer_time_dim - 1)
        file = os.path.join(path, filename)
        if not os.path.exists(file):
            if tracer_time_dim > 1:
                tracer_time_dim -= 1
            else:
                raise FileNotFoundError('No PETSc vectors found in {}.'.format(path))
        else:
            tracer_time_dim_found = True

    logging.debug('{} petsc vectors were found for each tracer.'.format(tracer_time_dim_found))

    ## calculate time_step, check time_dim_desired
    if time_dim_desired is not None:
        if tracer_time_dim % time_dim_desired == 0:
            time_step = int(tracer_time_dim / time_dim_desired)
        else:
            raise ValueError('The desired time dimension {0} can not be satisfied because the tracer time dimension {1} is not divisible by {0}.'.format(time_dim_desired, tracer_time_dim))
    else:
        time_dim_desired = tracer_time_dim
        time_step = 1

    assert tracer_time_dim % time_dim_desired == 0



    ## init trajectory
    if converted_result_shape is None:
        filename = METOS_TRAJECTORY_FILENAMES[0].format(0)
        file = os.path.join(path, filename)
        trajectory = util.petsc.universal.load_petsc_vec_to_numpy_array(file)
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
            for k in range(time_step):
                ## prepare filename
                file_nr = time_index * time_step + k
                filename = file_pattern.format(file_nr)
                file = os.path.join(path, filename)

                ## load vector and average
                vec = util.petsc.universal.load_petsc_vec_to_numpy_array(file)
                if set_negative_values_to_zero:
                    vec[vec < 0] = 0
                if k == 0:
                    trajectory_averaged = vec
                else:
                    trajectory_averaged += vec

            trajectory_averaged /= time_step

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




def load_trajectories_to_map(path, tracer_index, time_dim_desired=None):
    from ndop.model.constants import METOS_SPACE_DIM

    ## check input
    _check_tracer_index(tracer_index)

    ## load trajectory
    convert_function = lambda metos_vec: convert_metos_1D_to_3D(metos_vec)
    trajectory = load_trajectories_to_universal(path, convert_function=convert_function, converted_result_shape=METOS_SPACE_DIM, tracer_indices=tracer_index, time_dim_desired=time_dim_desired)
    trajectory = trajectory[0]

    assert trajectory.ndim == 4

    return trajectory



def load_trajectories_to_map_index_array(path, tracer_index, time_dim_desired=None):
    ## check input
    _check_tracer_index(tracer_index)

    ## load trajectory with universal function
    def convert_function(metos_vec):
        trajectory = convert_metos_1D_to_3D(metos_vec)
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
#         trajectory_point_array[t_index*point_len_per_t : (t_index+1)*point_len_per_t, 0] = t_index / t_dim
        trajectory_point_array[t_index*point_len_per_t : (t_index+1)*point_len_per_t, 0] = t_index
        trajectory_point_array[t_index*point_len_per_t : (t_index+1)*point_len_per_t, 1:] = trajectory[t_index]

#     ## convert metos indices to point values
#     trajectory_point_array[:, 1] = (trajectory_point_array[:, 1] + 0.5) * 360 / METOS_X_DIM
#     trajectory_point_array[:, 2] = (trajectory_point_array[:, 2] + 0.5) * 180 / METOS_Y_DIM - 90
#     trajectory_point_array[:, 3] = METOS_Z_CENTER[trajectory_point_array[:, 3].astype(dtype=np.int)]

    assert trajectory_point_array.ndim == 2
    assert trajectory_point_array.shape[1] == 5

    return trajectory_point_array

