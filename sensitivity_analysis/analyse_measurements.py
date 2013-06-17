def calculate_dop_measurement_error_variance(parameter_set = 1, t_len_desired = 1, debug_level = 0, required_debug_level = 1):
    from util import print_debug
    from NDOP_F import get_F, get_F_value
    from measurement_data import load_dop_measurements, load_dop_values
    
    print_debug(('Calculating DOP measurement error variance for parameter set ', parameter_set, ' with desired time-length of ', t_len_desired, '.'), debug_level, required_debug_level)
    
    measurements = load_dop_measurements(debug_level, required_debug_level + 1)
    measurement_values = load_dop_values(debug_level, required_debug_level + 1)
    
    vari = 0
    
    F = get_F(parameter_set, t_len_desired, debug_level, required_debug_level + 1)
    
    n_measurements = len(measurement_values)
    for i in range(n_measurements):
        F_i = get_F_value(F, 'dop', * measurements[i], debug_level = debug_level, required_debug_level = required_debug_level + 1)
        measurement_values_i = measurement_values[i]
        vari += (F_i - measurement_values_i) ** 2
    
    vari /= n_measurements
    
    return vari



def get_dop_measurement_error_variance(parameter_set = 1, t_len_desired = 1, debug_level = 0, required_debug_level = 1):
    from util import print_debug
    from NDOP_util import get_value
    from constants import oed_dop_vari_file_pattern
    
    print_debug(('Getting DOP measurement error variance for parameter set ', parameter_set, ' with desired time-length of ', t_len_desired, '.'), debug_level, required_debug_level)
    
    vari = get_value(calculate_dop_measurement_error_variance, oed_dop_vari_file_pattern, parameter_set, t_len_desired, debug_level, required_debug_level + 1)
    
    return vari



def calculate_po4_measurement_error_variance(parameter_set = 1, t_len_desired = 1, debug_level = 0, required_debug_level = 1):
    from util import print_debug
    from NDOP_F import get_F, get_F_value
    from measurement_data import load_po4_nobs, load_po4_values
    import numpy as np
    
    print_debug(('Calculating PO4 measurement error variance for parameter set ', parameter_set, ' with desired time-length of ', t_len_desired, '.'), debug_level, required_debug_level)
    
    nobs = load_po4_nobs(debug_level, required_debug_level + 1)
    measurement_values = load_po4_values(debug_level, required_debug_level + 1)
    
    vari = 0
    
    F = get_F(parameter_set, t_len_desired, debug_level, required_debug_level + 1)
    
    for multi_index in np.ndindex(* measurement_values.shape):
        nobs_i = nobs[multi_index]
        
        if nobs_i > 0:
            measurement_values_i = measurement_values[multi_index]
            F_i = F[1][multi_index]
            vari += nobs_i * ((F_i - measurement_values_i) ** 2)
    
    n_measurements = np.nansum(nobs)
    
    vari /= n_measurements
    
    return vari



def get_po4_measurement_error_variance(parameter_set = 1, t_len_desired = 1, debug_level = 0, required_debug_level = 1):
    from util import print_debug
    from NDOP_util import get_value
    from constants import oed_po4_vari_file_pattern
    
    print_debug(('Getting PO4 measurement error variance for parameter set ', parameter_set, ' with desired time-length of ', t_len_desired, '.'), debug_level, required_debug_level)
    
    vari = get_value(calculate_po4_measurement_error_variance, oed_po4_vari_file_pattern, parameter_set, t_len_desired, debug_level, required_debug_level + 1)
    
    return vari



def calculate_po4_measurement_error_variance_from_variance_data(debug_level = 0, required_debug_level = 1):
    from util import print_debug
    from measurement_data import load_po4_vari, load_po4_nobs
    import numpy as np
    
    print_debug(('Calculating averaged PO4 measurement error variance from variance data.'), debug_level, required_debug_level)
    
    vari = load_po4_vari(debug_level, required_debug_level + 1)
    nobs = load_po4_nobs(debug_level, required_debug_level + 1)
    
    averaged_vari = 0
    
    for multi_index in np.ndindex(* nobs.shape):
        nobs_i = nobs[multi_index]
        
        if nobs_i > 0:
            vari_i = vari[multi_index]
            averaged_vari += nobs_i * vari_i
    
    n_measurements = np.nansum(nobs)
    
    averaged_vari /= n_measurements
    
    return averaged_vari



def get_po4_measurement_error_variance_from_variance_data(debug_level = 0, required_debug_level = 1):
    from NDOP_util import get_value
    from util import print_debug
    from constants import oed_po4_vari_from_vari_data_file_pattern
    
    print_debug(('Getting averaged PO4 measurement error variance from variance data.'), debug_level, required_debug_level)
    
    calculate_function = lambda parameter_set, t_len_desired, debug_level, required_debug_level: calculate_po4_measurement_error_variance_from_variance_data(debug_level, required_debug_level)
    vari = get_value(calculate_function, oed_po4_vari_from_vari_data_file_pattern, None, None, debug_level, required_debug_level + 1)
    
    return vari