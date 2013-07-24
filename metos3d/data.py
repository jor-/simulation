import os
import math
import bisect
import numpy as np

import util.pattern

import util.interpolation
from util.debug import print_debug


def load_land_sea_mask(debug_level=0, required_debug_level=1):
    from ndop.metos3d.constants import METOS_LAND_SEA_MASK_FILE_PETSC, METOS_LAND_SEA_MASK_FILE_NPY
    
    try:
        land_sea_mask = np.load(METOS_LAND_SEA_MASK_FILE_NPY)
        
        print_debug('Returning land-sea-mask loaded from npy file', debug_level, required_debug_level, 'ndop.metos3d.data.load_land_sea_mask: ')
        
    except (OSError, IOError):
        import util.petsc
        
        land_sea_mask = util.petsc.load_petsc_mat(METOS_LAND_SEA_MASK_FILE_PETSC, dtype=int, debug_level=debug_level, required_debug_level=required_debug_level)
        land_sea_mask = land_sea_mask.transpose() # metos3d: x and y are changed
        
        print_debug(('Saving land-sea-mask to npy file ', METOS_LAND_SEA_MASK_FILE_NPY, '.'), debug_level, required_debug_level, 'ndop.metos3d.data.load_land_sea_mask: ')
        
        np.save(METOS_LAND_SEA_MASK_FILE_NPY, land_sea_mask)
        
        print_debug('Returning land-sea-mask loaded from petsc file', debug_level, required_debug_level, 'ndop.metos3d.data.load_land_sea_mask: ')
    
    return land_sea_mask



def convert_1D_to_3D(metos_vec, land_sea_mask, debug_level = 0, required_debug_level = 1):
    from ndop.metos3d.constants import METOS_Z_DIM
    
    x_dim, y_dim = land_sea_mask.shape
    
    # init array
    array = np.empty([x_dim, y_dim, METOS_Z_DIM], dtype=np.float64)
    array.fill(np.nan)
    
    # debug info
    print_debug(('Converting  metos ', metos_vec.shape, ' vector to ', array.shape, ' matrix.'), debug_level, required_debug_level, base_string='ndop.metos3d.data.convert_1D_to_3D: ')
    
    # fill array
    offset = 0
    for iy in range(y_dim):
        for ix in range(x_dim):
            length = land_sea_mask[ix, iy]
            if not length == 0:
                array[ix, iy, 0:length] = metos_vec[offset:offset+length]
                offset = offset + length
    
    return array
    
    

# def convert_1D_to_3D(metos_vec, land_sea_mask, debug_level = 0, required_debug_level = 1):
#     from ndop.metos3d.constants import METOS_Z_DIM
#     
#     
#     # metos3d: x and y are changed
#     x_dim, y_dim = land_sea_mask.shape
#     
#     # init array
#     array = np.empty([y_dim, x_dim, METOS_Z_DIM], dtype=np.float64)
#     array.fill(np.nan)
#     
#     # debug info
#     print_debug(('Converting  metos ', metos_vec.shape, ' vector to ', array.shape, ' matrix.'), debug_level, required_debug_level, base_string='ndop.metos3d.data.convert_1D_to_3D: ')
#     
#     # fill array
#     offset = 0
#     for ix in range(x_dim):
#         for iy in range(y_dim):
#             length = land_sea_mask[ix, iy]
#             if not length == 0:
#                 array[iy, ix, 0:length] = metos_vec[offset:offset+length]
#                 offset = offset + length
#     
#     return array



def load_trajectories(path, t_dim, time_step_size, land_sea_mask=None, debug_level = 0, required_debug_level = 1):
    import util.petsc
    from ndop.metos3d.constants import MODEL_TIME_STEP_SIZE_MAX, METOS_TRAJECTORY_FILENAMES
    
    number_of_petsc_vecs = MODEL_TIME_STEP_SIZE_MAX / time_step_size
    
    # check t_dim
    if number_of_petsc_vecs % t_dim == 0:
        t_step = int(number_of_petsc_vecs / t_dim)
    else:
        raise ValueError(concatenate_to_string(('The desired time dimension ', t_dim, ' can not be satisfied because ', number_of_petsc_vecs, ' is not divisible by ', t_dim, '.')))
    
    # init trajectory
    tracer_dim = len(METOS_TRAJECTORY_FILENAMES)
    filename = util.pattern.replace_int_pattern(METOS_TRAJECTORY_FILENAMES[0], 0)
    file = os.path.join(path, filename)
    trajectory = util.petsc.load_petsc_vec(file, debug_level = debug_level, required_debug_level = required_debug_level + 3)
    if land_sea_mask is not None:
        trajectory = convert_1D_to_3D(trajectory, land_sea_mask, debug_level = debug_level, required_debug_level = required_debug_level + 2)
    s_dim = trajectory.shape
    trajectory_shape = (tracer_dim, t_dim) + s_dim
    trajectory = np.zeros(trajectory_shape, dtype=np.float64)
    
    
    # debug info
    print_debug(('Loading trajectories from "', path, '" of size ', trajectory.shape, '.'), debug_level, required_debug_level, 'ndop.metos3d.data.load_trajectories: ')
    
    # load and calculate trajectory
    for i in range(tracer_dim):
        print_debug(('loading trajectory for tracer ', i), debug_level, required_debug_level + 1, 'ndop.metos3d.data.load_trajectories: ')
        file_pattern = METOS_TRAJECTORY_FILENAMES[i]
        for j in range(t_dim):
            print_debug(('loading trajectory for time ', j), debug_level, required_debug_level + 2, 'ndop.metos3d.data.load_trajectories: ')
            for k in range(t_step):
                file_nr = j * t_step + k
                filename = util.pattern.replace_int_pattern(file_pattern, file_nr)
                file = os.path.join(path, filename)
                if k == 0:
                    trajectory_averaged = util.petsc.load_petsc_vec(file)
                else:
                    trajectory_averaged += util.petsc.load_petsc_vec(file)
                
            trajectory_averaged /= t_step
            
            if land_sea_mask is not None:
                trajectory_averaged = convert_1D_to_3D(trajectory_averaged, land_sea_mask, debug_level=debug_level, required_debug_level=required_debug_level + 2)
            
            trajectory[i, j] = trajectory_averaged
        
    return trajectory



# def get_index_float(t, x, y, z, t_dim, debug_level = 0, required_debug_level = 1):
#     from ndop.metos3d.constants import METOS_X_RANGE, METOS_Y_RANGE, METOS_Z
#     
#     ## linear interpolate time, x and y index
#     x_index = (x - METOS_X_RANGE[0]) / (METOS_X_RANGE[1] - METOS_X_RANGE[0]) * (METOS_X_DIM)
#     y_index = (y - METOS_Y_RANGE[0]) / (METOS_Y_RANGE[1] - METOS_Y_RANGE[0]) * (METOS_Y_DIM)
#     t_index = (t - METOS_T_RANGE[0]) / (METOS_T_RANGE[1] - METOS_T_RANGE[0]) * (t_dim)
#     
#     ## lockup z layer
#     z_index_left = bisect.bisect(METOS_Z, z)
#     z_left = METOS_Z[z_index_left]
#     z_index_right = z_index_left + 1
#     z_right = METOS_Z[z_index_right]
#     
#     ## linear interpolate z index
#     if z_index_right < len(METOS_Z):
#         z_factor = (z_right - z) / (z_right - z_left)
#         z_index = z_factor * z_left + (1 - z_factor) * z_right
#     else:
#         z_index = z_left
#     
#     return (t_index, x_index, y_index, z_index)


def get_all_water_boxes(land_sea_mask, debug_level = 0, required_debug_level = 1):
    print_debug('Getting all water boxes.', debug_level, required_debug_level)
    
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



def get_nearest_water_box(land_sea_mask, x_index, y_index, z_index, debug_level = 0, required_debug_level = 1):
    print_debug('Getting nearest water box.', debug_level, required_debug_level)
    
    water_boxes = get_all_water_boxes(land_sea_mask, debug_level, required_debug_level + 1)
    
    index = [x_index, y_index, z_index]
    nearest_water_box = util.interpolation.get_nearest_value(water_boxes, index, debug_level, required_debug_level + 1)
    
    return nearest_water_box



def get_index(t, x, y, z, t_dim, land_sea_mask, debug_level = 0, required_debug_level = 1):
    from ndop.metos3d.constants import METOS_T_RANGE, METOS_X_RANGE, METOS_Y_RANGE, METOS_Z
    
    print_debug(('Getting nearest index for ', (t, x, y, z)), debug_level, required_debug_level)
    
    ## adjust x coordinates if negative
    if x < 0:
        x += 360
    
    ## check input
    if t < METOS_T_RANGE[0] or t > METOS_T_RANGE[1]:
        raise ValueError('Value "' + str(t) + '" of t is not in range "' + str(METOS_T_RANGE) + '".')
    if x < METOS_X_RANGE[0] or x > METOS_X_RANGE[1]:
        raise ValueError('Value "' + str(x) + '" of x is not in range "' + str(METOS_X_RANGE) + '".')
    if y < METOS_Y_RANGE[0] or y > METOS_Y_RANGE[1]:
        raise ValueError('Value "' + str(y) + '" of y is not in range "' + str(METOS_Y_RANGE) + '".')
    if z < METOS_Z[0]:
        raise ValueError('Value "' + str(z) + '" of z have ti be greater or equal to "' + str(METOS_Z[0]) + '".')
    
    ## linear interpolate time, x and y index
    (x_dim, y_dim) = land_sea_mask.shape
    def linear_interpolate(x, range, dim):
        i_float = (x - range[0]) / (range[1] - range[0]) * (dim)
        if i_float == dim:
            i_float = i_float - 1
        i_int = math.floor(i_float)
        return i_int, i_float
    
    (x_index, x_index_float) = linear_interpolate(x, METOS_X_RANGE, x_dim)
    (y_index, y_index_float) = linear_interpolate(y, METOS_Y_RANGE, y_dim)
    (t_index, _) = linear_interpolate(t, METOS_T_RANGE, t_dim)
    
    ## lockup z
    z_index = bisect.bisect(METOS_Z, z) - 1
    
    if z_index + 1 < len(METOS_Z):
        z_left = METOS_Z[z_index]
        z_right = METOS_Z[z_index + 1]
        
        z_index_float = z_index + (z - z_left) / (z_right - z_left)
    else:
        z_index_float = z_index
    
    
    print_debug(('Float indices for ', (t, x, y, z), ' are ', (t_index, x_index_float, y_index_float, z_index_float), '.'), debug_level, required_debug_level)
    
    ## get nearest water box if box is land
    box_value = land_sea_mask[x_index, y_index]
    if box_value is np.nan or box_value < z_index:
        print_debug(('Box ', (x_index, y_index, z_index), ' is land.'), debug_level, required_debug_level)
        (x_index, y_index, z_index) = get_nearest_water_box(land_sea_mask, x_index_float, y_index_float, z_index_float, debug_level, required_debug_level + 1)
    
    print_debug(('Nearest index for ', (x, y, z), ' is ', (x_index, y_index, z_index)), debug_level, required_debug_level)
    
    return (t_index, x_index, y_index, z_index)
    

    
#     ## linear interpolate z index
#     if z_index_right < len(METOS_Z):
#         z_factor = (z_right - z) / (z_right - z_left)
#         z_index = z_factor * z_left + (1 - z_factor) * z_right
#     else:
#         z_index = z_left
#     
#     (t_index, x_index, y_index, z_index) = get_index_float(t, x, y, z, t_dim, debug_level, required_debug_level + 1)
#     t_index = round(t_index)
#     (x_index, y_index, z_index) = get_nearest_water_box(x_index, y_index, z_index, debug_level, required_debug_level + 1)
#     
#     print_debug(('Nearest indix for ', (t, x, y, z,), ' is ', (t_index, x_index, y_index, z_index)), debug_level, required_debug_level)
#     
#     return (t_index, x_index, y_index, z_index)


# def get_index(t, x, y, z, t_dim, land_sea_mask, debug_level = 0, required_debug_level = 1):
#     print_debug(('Getting nearest indix for ', (t, x, y, z)), debug_level, required_debug_level)
#     
#     (t_index, x_index, y_index, z_index) = get_index_float(t, x, y, z, t_dim, debug_level, required_debug_level + 1)
#     t_index = round(t_index)
#     (x_index, y_index, z_index) = get_nearest_water_box(x_index, y_index, z_index, debug_level, required_debug_level + 1)
#     
#     print_debug(('Nearest indix for ', (t, x, y, z,), ' is ', (t_index, x_index, y_index, z_index)), debug_level, required_debug_level)
#     
#     return (t_index, x_index, y_index, z_index)