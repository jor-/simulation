def get_file(file_pattern, parameter_set = 1, t_len_desired = 1, debug_level = 0, required_debug_level = 1):
    from constants import oed_parameter_set_pattern, oed_time_length_pattern
    
    file = file_pattern.replace(oed_parameter_set_pattern, str(parameter_set));
    file = file.replace(oed_time_length_pattern, str(t_len_desired));
    
    return file



def save_value(calculate_method, file_pattern, parameter_set = 1, t_len_desired = 1, debug_level = 0, required_debug_level = 1):
    from util import print_debug, make_dir_if_not_exists
    import numpy as np
    
    file = get_file(file_pattern, parameter_set, t_len_desired, debug_level, required_debug_level + 1)
    
    print_debug(('Saving value for parameter set ', parameter_set, ' with desired time-length of ', t_len_desired, ' at ', file, '.'), debug_level, required_debug_level)
    
    value = calculate_method(parameter_set, t_len_desired, debug_level = debug_level, required_debug_level = required_debug_level + 1)
    
    make_dir_if_not_exists(file, debug_level, required_debug_level + 1)
    np.save(file, value)
    
    print_debug(('Value for parameter set ', parameter_set, ' with desired time-length of ', t_len_desired, ' saved at ', file, '.'), debug_level, required_debug_level)
    
    return value



def get_value(calculate_method, file_pattern, parameter_set = 1, t_len_desired = 1, debug_level = 0, required_debug_level = 1):
    from util import print_debug
    import numpy as np
    
    file = get_file(file_pattern, parameter_set, t_len_desired, debug_level, required_debug_level + 1)
    try:
        value = np.load(file, 'r')
    except IOError:
        print_debug(('File ', file, ' for value for parameter set ', parameter_set, ' with desired time-length of ', t_len_desired, ' not available.'), debug_level, required_debug_level)
        value = save_value(calculate_method, file_pattern, parameter_set, t_len_desired, debug_level, required_debug_level + 1)
    
    print_debug(('Value for parameter set ', parameter_set, ' with desired time-length of ', t_len_desired, ' loaded from ', file, '.'), debug_level, required_debug_level)
    
    return value