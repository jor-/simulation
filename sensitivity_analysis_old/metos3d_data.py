def load_metos_land_sea_mask_petsc(debug_level = 0, required_debug_level = 1):
    from util import load_petsc_mat, print_debug
    from constants import metos_land_sea_mask_petsc
    
    print_debug(('Loading metos land-sea-mask from petsc file.'), debug_level, required_debug_level)
    
    lsm = load_petsc_mat(metos_land_sea_mask_petsc, int, debug_level, required_debug_level)
    
    return lsm


def load_metos_land_sea_mask(debug_level = 0, required_debug_level = 1):
    from util import make_dir_if_not_exists, print_debug
    from constants import metos_land_sea_mask_npy
    import numpy as np
    
    print_debug(('Loading metos land-sea-mask from npy file.'), debug_level, required_debug_level)
    
    try:
        lsm = np.load(metos_land_sea_mask_npy, 'r')
    except IOError:
        lsm = load_metos_land_sea_mask_petsc(debug_level, required_debug_level + 1)
        make_dir_if_not_exists(metos_land_sea_mask_npy, debug_level, required_debug_level + 1)
        np.save(metos_land_sea_mask_npy, lsm)
    
    return lsm



def load_metos_h(file_pattern, derivative_number, debug_level = 0, required_debug_level = 1):
    from util import print_debug
    from constants import metos_derivative_number_pattern
    
    filename = file_pattern.replace(metos_derivative_number_pattern, '{:+d}'.format(derivative_number))
    
    print_debug(('Loading h file ', filename, '.'), debug_level, required_debug_level)
    
    file = open(filename, 'r')
    h = float(file.readline())
    file.close()        
    
    return h


    
def load_metos_trajectory(file_pattern, t_len_desired, debug_level = 0, required_debug_level = 1):
    from util import load_petsc_vec, concatenate_to_string, print_debug
    from constants import metos_time_pattern, metos_time_pattern_length_str, metos_t_length
    import numpy as np
    
    # check t_len_desired
    if metos_t_length % t_len_desired == 0:
        t_step = int(metos_t_length / t_len_desired)
    else:
        raise ValueError(concatenate_to_string(('The desired time dimension ', t_len_desired, ' can not be satisfied because it is not a multiple of ', metos_t_length, '.')))
    
    # init trajectory
    file = file_pattern.replace(metos_time_pattern, str(0).zfill(metos_time_pattern_length_str))
    s_len = load_petsc_vec(file, debug_level = debug_level, required_debug_level = required_debug_level + 2).size
    trajectory = np.zeros([t_len_desired, s_len], dtype=np.float64)
    
    # debug info
    print_debug(('Loading trajectory array ', file_pattern, ' of size ', trajectory.shape, '.'), debug_level, required_debug_level)
    
    # load and calculate trajectory
    for i in range(t_len_desired):
        print_debug(('loading trajectory for time ', i), debug_level, required_debug_level + 1)
        for j in range(t_step):
            file_nr = i * t_step + j
            file = file_pattern.replace(metos_time_pattern, str(file_nr).zfill(metos_time_pattern_length_str))
            trajectory[i] += load_petsc_vec(file, debug_level = debug_level, required_debug_level = required_debug_level + 2)
            
        trajectory[i] /= t_step
        
    return trajectory



def convert_metos_1D_to_3D(metos_vec, land_sea_mask, debug_level = 0, required_debug_level = 1):
    from util import print_debug
    from constants import metos_z
    import numpy as np
    
    nx, ny = land_sea_mask.shape
    nz = len(metos_z)
    
    # init array
    array = np.empty([nz, nx, ny], dtype=np.float64)
    array.fill(np.nan)
    
    # debug info
    print_debug(('Converting  metos ', metos_vec.shape, ' vector to ', array.shape, ' matrix.'), debug_level, required_debug_level)
    
    # fill array
    offset = 0
    for ix in range(nx):
        for iy in range(ny):
            length = land_sea_mask[ix, iy]
            if not length == 0:
                array[0:length, ix, iy] = metos_vec[offset:offset+length]
                offset = offset + length
    
    return array



def get_all_water_boxes(debug_level = 0, required_debug_level = 1):
    from util import print_debug
    import numpy as np
    
    print_debug(('Getting all water boxes.'), debug_level, required_debug_level)
    
    land_sea_mask =  load_metos_land_sea_mask(debug_level, required_debug_level + 1)
    
    (water_y, water_x) = np.where(land_sea_mask != 0)
    water_len = np.sum(land_sea_mask)
    water_boxes = np.empty([water_len, 3], dtype=np.int)
    
    j = 0
    for i in range(len(water_x)):
        x_i = water_x[i]
        y_i = water_y[i]
        z_i = land_sea_mask[y_i, x_i]
        
        j_range = range(j, j + z_i)
        water_boxes[j_range, 0] = x_i
        water_boxes[j_range, 1] = y_i
        water_boxes[j_range, 2] = range(z_i)
        
        j += z_i
    
    return water_boxes



def get_nearest_water_box(x_index, y_index, z_index, debug_level = 0, required_debug_level = 1):
    from util import get_nearest_value, print_debug
    
    print_debug(('Getting nearest water box.'), debug_level, required_debug_level)
    
    water_boxes = get_all_water_boxes()
    
    index = [x_index, y_index, z_index]
    nearest_water_box = get_nearest_value(water_boxes, index, debug_level, required_debug_level + 1)
    
    return nearest_water_box
    