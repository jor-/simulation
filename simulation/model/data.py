import os
import itertools

import numpy as np

import simulation.model.constants

import util.petsc.universal
import util.logging
logger = util.logging.logger


## convert Metos vector to 3D vector

def convert_metos_1D_to_3D(metos_vec):
    assert len(metos_vec) == simulation.model.constants.METOS_VECTOR_LEN
    
    METOS_LSM = simulation.model.constants.METOS_LSM

    ## init array
    array = np.empty([METOS_LSM.x_dim, METOS_LSM.y_dim, METOS_LSM.z_dim], dtype=np.float64)
    array.fill(np.nan)

    ## fill array
    logger.debug('Converting metos {} vector to {} matrix.'.format(metos_vec.shape, array.shape))

    offset = 0
    for iy in range(METOS_LSM.y_dim):
        for ix in range(METOS_LSM.x_dim):
            length = METOS_LSM[ix, iy]
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

def load_trajectories_to_universal(path, convert_function=None, converted_result_shape=None, tracers=None, time_dim_desired=None, set_negative_values_to_zero=False):
    logger.debug('Loading trajectories with tracers {}, desired time dim {}, set_negative_values_to_zero {} and convert function {} with result shape {} from {}.'.format(tracers, time_dim_desired, set_negative_values_to_zero, convert_function, converted_result_shape, path))

    ## check input
    if isinstance(tracers, str):
        tracers = [tracers]

    # check convert_function
    if convert_function is None:
        convert_function = lambda x: x
        if converted_result_shape is not None:
            raise ValueError('The convert function is None but the converted result shape is not None ({}).'.format(converted_result_shape))
    elif not callable(convert_function):
        raise ValueError('The convert function {} has to be callable.'.format(convert_function))

    assert callable(convert_function)

    ## calculate tracer_time_dim
    tracer_time_dim = simulation.model.constants.METOS_T_DIM
    tracer_time_dim_found = False
    while not tracer_time_dim_found:
        filename = simulation.model.constants.METOS_TRAJECTORY_FILENAME.format(tracer=tracers[0], time_step=tracer_time_dim - 1)
        file = os.path.join(path, filename)
        if not os.path.exists(file):
            if tracer_time_dim > 1:
                tracer_time_dim -= 1
            else:
                raise FileNotFoundError('No PETSc vectors found in {}.'.format(path))
        else:
            tracer_time_dim_found = True

    logger.debug('{} petsc vectors were found for each tracer.'.format(tracer_time_dim_found))

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
        filename = simulation.model.constants.METOS_TRAJECTORY_FILENAME.format(tracer=tracers[0], time_step=0)
        file = os.path.join(path, filename)
        trajectory = util.petsc.universal.load_petsc_vec_to_numpy_array(file)
        converted_result_shape = convert_function(trajectory).shape

    tracers_len = len(tracers)
    trajectory_shape = (tracers_len, time_dim_desired) + converted_result_shape
    trajectory = np.zeros(trajectory_shape, dtype=np.float64)

    ## load and calculate trajectory
    logger.debug('Loading trajectories from {} to array of size {}.'.format(path, trajectory.shape))

    for tracers_index in range(tracers_len):
        tracer = tracers[tracers_index]

        logger.debug('Loading trajectory for tracer {}.'.format(tracer))
        for time_index in range(time_dim_desired):
            ## average trajectory
            for k in range(time_step):
                ## prepare filename
                file_nr = time_index * time_step + k
                filename = simulation.model.constants.METOS_TRAJECTORY_FILENAME.format(tracer=tracer, time_step=file_nr)
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

            trajectory[tracers_index, time_index] = trajectory_averaged

    logger.debug('Trajectory with shape {} loaded.'.format(trajectory.shape))

    return trajectory


def load_trajectories_to_map(path, tracers, time_dim_desired=None):
    ## load trajectory
    convert_function = lambda metos_vec: convert_metos_1D_to_3D(metos_vec)
    trajectory = load_trajectories_to_universal(path, convert_function=convert_function, converted_result_shape=simulation.model.constants.METOS_SPACE_DIM, tracers=tracers, time_dim_desired=time_dim_desired)
    trajectory = trajectory[0]

    assert trajectory.ndim == 4
    return trajectory


def load_trajectories_to_map_index_array(path, tracers, time_dim_desired=None):
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

    trajectory = load_trajectories_to_universal(path, convert_function=convert_function, tracers=tracers, time_dim_desired=time_dim_desired)
    trajectory = trajectory[0]
    assert trajectory.ndim == 3
    assert np.all(trajectory[:, :, :3] % 1 == 0)

    ## convert time index to point value
    t_dim, point_len_per_t, point_dim  = trajectory.shape
    trajectory_point_array = np.empty((t_dim * point_len_per_t, point_dim + 1))
    for t_index in range(t_dim):
        trajectory_point_array[t_index*point_len_per_t : (t_index+1)*point_len_per_t, 0] = t_index
        trajectory_point_array[t_index*point_len_per_t : (t_index+1)*point_len_per_t, 1:] = trajectory[t_index]

    assert trajectory_point_array.ndim == 2
    assert trajectory_point_array.shape[1] == 5
    return trajectory_point_array

