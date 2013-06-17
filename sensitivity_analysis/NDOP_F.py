def calculate_F(parameter_set = 1, t_len_desired = 1, debug_level = 0, required_debug_level = 1):
    from util import print_debug
    from metos3d_data import load_metos_land_sea_mask, load_metos_trajectory, convert_metos_1D_to_3D
    from constants import metos_trajectory_file_pattern, metos_tracer_pattern, metos_parameter_set_pattern, metos_tracers, metos_derivative_number_pattern, metos_t_length, metos_z
    import numpy as np
    
    print_debug(('Calculating model output for parameter set ', parameter_set, ' with desired time-length of ', t_len_desired, '.'), debug_level, required_debug_level)
    
    land_sea_mask = load_metos_land_sea_mask(debug_level, required_debug_level + 1)
    
    t_len = metos_t_length
    x_len, y_len = land_sea_mask.shape
    z_len = len(metos_z)
    
    n_metos_tracers = len(metos_tracers)
    
    F = np.empty([n_metos_tracers, t_len_desired, z_len, x_len, y_len], dtype=np.float64)
    
    metos_trajectory_file_pattern = metos_trajectory_file_pattern.replace(metos_parameter_set_pattern, str(parameter_set))
    metos_trajectory_file_pattern = metos_trajectory_file_pattern.replace(metos_derivative_number_pattern, '+0')
    
    for i in range(n_metos_tracers):
        current_trajectory_file_pattern = metos_trajectory_file_pattern.replace(metos_tracer_pattern, metos_tracers[i])
        metos_F = load_metos_trajectory(current_trajectory_file_pattern, t_len_desired, debug_level, required_debug_level + 1)
        for k in range(t_len_desired):
            F[i,k,:,:,:] = convert_metos_1D_to_3D(metos_F[k], land_sea_mask, debug_level, required_debug_level + 2)
        
    return F



def get_F(parameter_set = 1, t_len_desired = 1, debug_level = 0, required_debug_level = 1):
    from util import print_debug
    from NDOP_util import get_value
    from constants import oed_F_file_pattern
    
    print_debug(('Getting F value for parameter set ', parameter_set, ' with desired time-length of ', t_len_desired, '.'), debug_level, required_debug_level)
    
    F = get_value(calculate_F, oed_F_file_pattern, parameter_set, t_len_desired, debug_level, required_debug_level + 1)
    
    return F



def get_F_value(F, tracer, time, x, y, z, debug_level = 0, required_debug_level = 1):
    # 0 <= time < 1
    # -180 <= x < 180
    # -90 <= y < 90
    # 0 <= z <= 5200
    from util import linear_interpolate_array, print_debug
    from metos3d_data import get_nearest_water_box
    from constants import metos_tracers, metos_z
    import numpy as np
    import bisect
    
    # debug informations
    print_debug(('Getting F value for index ', (tracer, time, x, y, z), '.'), debug_level, required_debug_level)
    
    # get dimensions
    (F_tracer_len, F_time_len, F_z_len, F_y_len, F_x_len) = F.shape[0:5]
    
    # lockup tracer index if tracer is a string
    if isinstance(tracer, str):
        tracer_index = metos_tracers.index(tracer.lower())
    else:
        tracer_index = tracer
    
    time_index = time * (F_time_len - 1)
    
    if x < 0:
        x = x + 360
    
    # x in F from 0 to 360
    x_index = x / 360 * (F_x_len - 1)
    
    # y in F from -90 to +90
    y_index = (y + 90) / 180 * (F_y_len - 1)
    
    # lockup z layer
    z_index_left = bisect.bisect(metos_z, z)
    z_left = metos_z[z_index_left]
    z_index_right = z_index_left + 1
    z_right = metos_z[z_index_right]
    
    if z_index_right < len(metos_z):
        z_factor = (z_right - z) / (z_right - z_left)
        z_index = z_factor * z_left + (1 - z_factor) * z_right
    else:
        z_index = z_left
    
    # get F value
#     try:
    F_value = linear_interpolate_array(F, [tracer_index, time_index, z_index, y_index, x_index], True, debug_level, required_debug_level + 1)
#     except ValueError:
#         (x_nearest_index, y_nearest_index, z_nearest_index) = get_nearest_water_box(x_index, y_index, z_index, debug_level, required_debug_level + 1)
#         F_value = F[tracer_index, time_index, z_nearest_index, y_nearest_index, x_nearest_index]
    
    # debug informations
    print_debug(('F value for index ', (tracer, time, x, y, z), ' is ', F_value, '.'), debug_level, required_debug_level)
    
    if np.any(np.isnan(F_value)):
        raise ValueError('NAN in F_value!')
    
    return F_value
