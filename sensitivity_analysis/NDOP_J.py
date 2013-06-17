def calculate_metos_derivative(trajectory_file_pattern, h_file_pattern, derivative_number, t_len_desired, debug_level = 0, required_debug_level = 1):  
    from util import print_debug
    from metos3d_data import load_metos_trajectory, load_metos_h
    from constants import metos_derivative_number_pattern
    import numpy as np
    
    print_debug(('Calculating  derivative ', derivative_number, '.'), debug_level, required_debug_level)
    
    trajectory_file_pattern_pos = trajectory_file_pattern.replace(metos_derivative_number_pattern, '{:+d}'.format(derivative_number))
    derivative = load_metos_trajectory(trajectory_file_pattern_pos, t_len_desired, debug_level, required_debug_level + 1)
    h_pos = load_metos_h(h_file_pattern, derivative_number, debug_level, required_debug_level + 1) 
      
    trajectory_file_pattern_neg = trajectory_file_pattern.replace(metos_derivative_number_pattern, '{:+d}'.format(- derivative_number))
    tmp = load_metos_trajectory(trajectory_file_pattern_neg, t_len_desired, debug_level, required_debug_level + 1)
    h_neg = load_metos_h(h_file_pattern, - derivative_number, debug_level, required_debug_level + 1)
    
    h2 = h_pos + h_neg
    derivative = (derivative / h2) - (tmp / h2)
    
    return derivative
    


def calculate_J(parameter_set = 1, t_len_desired = 1, debug_level = 0, required_debug_level = 1):
    from util import print_debug
    from metos3d_data import load_metos_land_sea_mask, convert_metos_1D_to_3D
    from constants import metos_trajectory_file_pattern, metos_h_file_pattern, metos_tracer_pattern, metos_parameter_set_pattern, metos_tracers, metos_derivative_length, metos_t_length, metos_z
    import numpy as np
    
    print_debug(('Calculating Jacobi matrix for parameter set ', parameter_set, ' with desired time-length of ', t_len_desired), debug_level, required_debug_level)
    
    land_sea_mask = load_metos_land_sea_mask(debug_level, required_debug_level + 1)
    
    t_len = metos_t_length
    x_len, y_len = land_sea_mask.shape
    z_len = len(metos_z)
    
    n_metos_tracers = len(metos_tracers)
    n_derivative_numbers = metos_derivative_length
    
    J = np.empty([n_metos_tracers, t_len_desired, z_len, x_len, y_len, n_derivative_numbers], dtype=np.float64)
    
    metos_trajectory_file_pattern = metos_trajectory_file_pattern.replace(metos_parameter_set_pattern, str(parameter_set)) 
    metos_h_file_pattern = metos_h_file_pattern.replace(metos_parameter_set_pattern, str(parameter_set))
    
    for i in range(n_metos_tracers):
        current_trajectory_file_pattern = metos_trajectory_file_pattern.replace(metos_tracer_pattern, metos_tracers[i]) 
        for j in range(n_derivative_numbers):
            metos_derivative = calculate_metos_derivative(current_trajectory_file_pattern, metos_h_file_pattern, j + 1, t_len_desired, debug_level, required_debug_level + 1)
            for k in range(t_len_desired):
                J[i,k,:,:,:,j] = convert_metos_1D_to_3D(metos_derivative[k], land_sea_mask, debug_level, required_debug_level + 2)
        
    return J



def get_J(parameter_set = 1, t_len_desired = 1, debug_level = 0, required_debug_level = 1):
    from NDOP_util import get_value
    from constants import oed_J_file_pattern
    
    J = get_value(calculate_J, oed_J_file_pattern, parameter_set, t_len_desired, debug_level, required_debug_level)
    return J



def get_J_value(J, tracer, time, x, y, z, debug_level = 0, required_debug_level = 1):
    from NDOP_F import get_F_value
    
    J_value = get_F_value(J, tracer, time, x, y, z, debug_level, required_debug_level)
    return J_value