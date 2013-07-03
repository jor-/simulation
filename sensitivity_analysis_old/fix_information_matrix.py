def calculate_fix_FIM_PO4(parameter_set = 1, debug_level = 0, required_debug_level = 1):
    from NDOP_J import get_J
    from measurement_data import load_po4_nobs, load_po4_vari
    from util import print_debug
    import numpy as np
    
    print_debug(('calculating Fisher-information matrix for PO4 for parameter set ', parameter_set), debug_level, required_debug_level)
    
    PO4_nobs = load_po4_nobs(debug_level, required_debug_level + 1)
    PO4_vari = load_po4_vari(debug_level, required_debug_level + 1)
    
    t_dim = PO4_nobs.shape[0]
    J = get_J(parameter_set, t_dim, debug_level, required_debug_level + 1)
    
    PO4_shape = J[1].shape
    der_len = PO4_shape[4]
    
    FIM = np.zeros([der_len, der_len], dtype=np.float64)
    
    for multi_index in np.ndindex(*PO4_shape[0:4]):
        nobs = PO4_nobs[multi_index]
        if nobs > 0:
            vari = PO4_vari[multi_index]
            FIM_vec = J[1][multi_index]
            # check for nan
            if not np.isnan(np.sum(FIM_vec) + vari):
                FIM += np.outer(FIM_vec, FIM_vec) * nobs / vari
            else:
                raise ValueError('NAN in FIM_vec')

    return FIM



def calculate_fix_FIM_DOP(parameter_set = 1, debug_level = 0, required_debug_level = 1):
    from NDOP_J import get_J, get_J_value
    from measurement_data import load_dop_measurements, load_dop_values
    from analyse_measurements import get_dop_measurement_error_variance
    from util import print_debug
    import numpy as np
    
    print_debug(('calculating Fisher-information matrix for DOP for parameter set ', parameter_set), debug_level, required_debug_level)
    
    measurements = load_dop_measurements(debug_level, required_debug_level + 1)
    measurement_values = load_dop_values(debug_level, required_debug_level + 1)
    
    t_dim = 12
    J = get_J(parameter_set, t_dim, debug_level, required_debug_level + 1)
    
    J_tracer_shape = J[0].shape
    der_len = J_tracer_shape[4]
    FIM = np.zeros([der_len, der_len], dtype=np.float64)
    
    tracer = 'dop'
    n_measurements = len(measurement_values)
    for i in range(n_measurements):
        gradient = get_J_value(J, tracer, * measurements[i], debug_level = debug_level, required_debug_level = required_debug_level + 1)
        
        # check for nan
        if not np.isnan(np.sum(gradient)):
            FIM += np.outer(gradient, gradient)
        else:
            raise ValueError('NAN in FIM_vec')
    
    vari = get_dop_measurement_error_variance(parameter_set, t_dim, debug_level, required_debug_level + 1)
    FIM /= vari

    return FIM



def get_fix_FIM_tracer(tracer, parameter_set = 1, debug_level = 0, required_debug_level = 1):
    from NDOP_util import get_value
    from constants import oed_fix_FIM_tracer_file_pattern, oed_tracer_pattern
    from util import print_debug, concatenate_to_string
    
    print_debug(('Getting fix information matrix for tracer ', tracer, ' and parameter set ', parameter_set, '.'), debug_level, required_debug_level)
    
    if tracer.lower() == 'dop':
        calculate_function = lambda parameter_set, t_len_desired, debug_level, required_debug_level: calculate_fix_FIM_DOP(parameter_set, debug_level, required_debug_level)
    elif tracer.lower() == 'po4':
        calculate_function = lambda parameter_set, t_len_desired, debug_level, required_debug_level: calculate_fix_FIM_PO4(parameter_set, debug_level, required_debug_level)
    else:
        raise ValueError(concatenate_to_string('unknown tracer: ', tracer))
        
    oed_fix_FIM_tracer_file_pattern = oed_fix_FIM_tracer_file_pattern.replace(oed_tracer_pattern, tracer);
    FIM = get_value(calculate_function, oed_fix_FIM_tracer_file_pattern, parameter_set, None, debug_level, required_debug_level + 1)
    
    return FIM


# def get_fix_FIM(parameter_set = 1, debug_level = 0, required_debug_level = 1):
#     from util import print_debug
#     from NDOP_util import get_value
#     from constants import oed_FIM_file_pattern
#     
#     print_debug(('Getting fix information matrix for parameter set ', parameter_set, '.'), debug_level, required_debug_level)
#     
#     calculate_function = lambda parameter_set, t_len_desired, debug_level, required_debug_level: calculate_fix_FIM_PO4(parameter_set, debug_level, required_debug_level)
#     FIM = get_value(calculate_function, oed_FIM_file_pattern, parameter_set, 1, debug_level, required_debug_level + 1)
#     
#     return F





# def get_fix_FIM_file(parameter_set = 1, debug_level = 0, required_debug_level = 1):
#     from constants import oed_FIM_file_pattern, oed_parameter_set_pattern
#     
#     FIM_file = oed_FIM_file_pattern.replace(oed_parameter_set_pattern, parameter_set);
#     
#     return FIM_file
# 
# 
# def save_fix_FIM(parameter_set = 1, debug_level = 0, required_debug_level = 1):
#     from util import print_debug
#     import numpy as np
#     
#     FIM_file = get_fix_FIM_file(parameter_set, debug_level, required_debug_level + 1);
#     
#     print_debug(('saving Fisher-information matrix for parameter set ', parameter_set, ' at ', FIM_file, '.'), debug_level, required_debug_level)
#     
#     FIM = calculate_fix_FIM(debug_level, required_debug_level + 1)
#     np.save(FIM_file, FIM)
#     
#     print_debug(('Fisher-information matrix for parameter set ', parameter_set, ' saved at ', FIM_file, '.'), debug_level, required_debug_level)
#     
#     return FIM
# 
# 
# 
# def get_fix_FIM(parameter_set = 1, debug_level = 0, required_debug_level = 1):
#     from util import print_debug
#     import numpy as np
#     
#     FIM_file = get_fix_FIM_file(parameter_set, debug_level, required_debug_level + 1);
#     
#     try:
#         FIM = np.load(FIM_file, 'r')
#     except IOError:
#         print_debug(('File ', FIM_file, ' for Fisher-information matrix not available.'), debug_level, required_debug_level)
#         FIM = save_fix_FIM(debug_level, required_debug_level + 1)
#     
#     print_debug(('Fisher-information matrix for parameter set ', parameter_set, ' loaded from ', FIM_file, '.'), debug_level, required_debug_level)
#     
#     return FIM



def get_confidence_factors(FIM, alpha = 0.95, debug_level = 0, required_debug_level = 1):
    from numpy.linalg import inv
    from numpy import diag
    from scipy.stats import chi2
    
    C = inv(FIM)
    d = diag(C)
    
    n = C.shape[0]
    
    # calculate chi-square quantil with confidence level alpha and n degrees of freedom
    gamma = chi2.ppf(alpha, n)
    
    confidence_factors = d**(1/2) * gamma**(1/2)
    
    return confidence_factors