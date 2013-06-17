import pdb
import analyse_measurements as ana
import fix_information_matrix as im
import sys
from numpy import linalg as la

args = sys.argv
options = {'parameter_set': int(args[1]), 't_len_desired': int(args[2]), 'debug_level': int(args[3])}

dop_vari = ana.get_dop_measurement_error_variance(**options)
po4_vari = ana.get_po4_measurement_error_variance(**options)
averaged_po4_vari = ana.get_po4_measurement_error_variance_from_variance_data(options['debug_level'])

FIM_dop = im.get_fix_FIM_tracer('dop', options['parameter_set'], options['debug_level'])
FIM_po4 = im.get_fix_FIM_tracer('po4', options['parameter_set'], options['debug_level'])
FIM = FIM_dop + FIM_po4

C_dop = la.inv(FIM_dop)
C_po4 = la.inv(FIM_po4)
C = la.inv(FIM)

CF_dop = im.get_confidence_factors(FIM_dop)
CF_po4 = im.get_confidence_factors(FIM_po4)
CF = im.get_confidence_factors(FIM)

print('Options:')
print(options)

print('DOP measurement error variance (estimated with model output):')
print(dop_vari)
print('PO4 measurement error variance (estimated with model output):')
print(po4_vari)
print('PO4 measurement error variance (averaged from variance data):')
print(averaged_po4_vari)

# print('Covariance matrix DOP')
# print(C_dop)
# print('Covariance matrix PO4')
# print(C_po4)
# print('Covariance matrix')
# print(C)

print('Confidence factors DOP')
print(CF_dop)
print('Confidence factors PO4')
print(CF_po4)
print('Confidence factors together')
print(CF)