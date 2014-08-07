import numpy as np

from ndop.accuracy.asymptotic import OLS, WLS, GLS, Family
from util.logging import Logger

# def calculate(p, data_kind, opt_class):
#     print('Information matrix for data kind {} and opt kind {}:'.format(data_kind, opt_class.__name__))
#     
#     information_matrix_object = opt_class(data_kind, 10000)
#     information_matrix = information_matrix_object.information_matrix(p)
#     
#     print(information_matrix)
#     
# 
# with Logger():
#     p = np.loadtxt('/work_O2/sunip229/NDOP/model_output/time_step_0001/parameter_set_00002/parameters.txt')
#     
# #     for opt_class in (GLS, WLS, OLS):
# #         calculate(p, 'WOD', opt_class)
# #     for opt_class in (WLS, OLS):
# #         calculate(p, 'WOA', opt_class)
#     
#     for opt_class in (OLS, WLS):
#         calculate(p, 'WOA', opt_class)
#     for opt_class in (OLS, WLS, GLS):
#         calculate(p, 'WOD', opt_class)
#     
#     print('finished')


with Logger():
    p = np.loadtxt('/work_O2/sunip229/NDOP/model_output/time_step_0001/parameter_set_00184/parameters.txt')
    
#     for data_kind in ('WOA', 'WOD'):
#         family = Family(OLS, data_kind, 10000)
#         family.parameter_confidence(p)
#     
#     for data_kind in ('WOA', 'WOD'):
#         family = Family(OLS, data_kind, 10000)
#         family.average_model_confidence(p)
#     
#     for data_kind in ('WOA', 'WOD'):
#         family = Family(OLS, data_kind, 10000)
#         family.average_model_confidence_increase(p)
        
    for data_kind in ('WOA', 'WOD'):
        family = Family(OLS, data_kind, 10000)
#         family.parameter_confidence(p)
#         family.average_model_confidence(p)
        family.average_model_confidence_increase(p)
    
    print('finished')