import sys
import argparse

import os
import numpy as np

import util.pattern
import util.io.fs

import ndop.model.eval
from ndop.optimization.matlab.constants import GLS_DICT


COST_FUNCTION_NAMES = ['{data_kind}/{cost_function}'.format(data_kind=dk, cost_function=cf) for dk in ('WOA', 'WOD', 'WOD_TMM_1', 'WOD_TMM_0') for cf in ('OLS', 'WLS', 'LWLS')] + ['{data_kind}/GLS/min_values_{min_values}/max_year_diff_inf/min_diag_1e-02'.format(data_kind=dk.replace('.', '_TMM_'), min_values=mv) for dk in GLS_DICT.keys() for mv in GLS_DICT[dk]]
COST_FUNCTION_NAMES.sort()

COST_FUNCTION_OUTPUT_DIRNAME = 'cost_functions'
COST_FUNCTION_F_FILENAME = 'f.txt'


def values(parameter_set_index, cost_function_names):
    m = ndop.model.eval.Model()
    parameter_set_dir = m.parameter_set_dir_with_index(parameter_set_index)
    
    values = {}

    for cost_function_name in cost_function_names:
        cost_function_f_file = os.path.join(parameter_set_dir, COST_FUNCTION_OUTPUT_DIRNAME, cost_function_name, COST_FUNCTION_F_FILENAME)
        try:
            value = np.loadtxt(cost_function_f_file)
        except FileNotFoundError:
            pass
        else:
            values[cost_function_name] = value
    
    return values


def min_values(cost_function_names):
    ## init dicts
    best_values = {}
    for cost_function_name in cost_function_names:
        best_values[cost_function_name] = np.inf
    best_indices = {}
    for cost_function_name in cost_function_names:
        best_indices[cost_function_name] = -1
    
    ## check alls parameter sets
    m = ndop.model.eval.Model()
    used_indices = m._parameter_db.used_indices()
    for parameter_set_index in used_indices:
        values_dict = values(parameter_set_index, cost_function_names)
        for key, value in values_dict.items():
            if value < best_values[key]:
                best_values[key] = value
                best_indices[key] = parameter_set_index
                
    return (best_indices, best_values)


def print_min_values(cost_function_names):
    (best_indices, best_values) = min_values(cost_function_names)
    m = ndop.model.eval.Model()
    for cost_function_name in best_indices.keys():
        
        p = m._parameter_db.get_value(best_indices[cost_function_name])
        p_str = np.array_str(p, precision=2)
        p_str = p_str.replace('\n', '').replace('\r', '')
        
        print('For {} parameter set has {} the min value {} with parameters:'.format(cost_function_name, best_indices[cost_function_name], best_values[cost_function_name]))
        print('{}'.format(p_str))


def all_values_for_min_values(cost_function_names):
    (best_indices, best_values) = min_values(cost_function_names)
    all_values = {}
    for parameter_set_index in best_indices.values():
        all_values[parameter_set_index] = values(parameter_set_index, cost_function_names)
    return all_values


def print_all_values_for_min_values(cost_function_names):
    print('All min cost function values for {}:'.format(cost_function_names))
    all_values = all_values_for_min_values(cost_function_names)
    for parameter_set_index, values_dict in all_values.items():
        values = np.array(list(values_dict.values()))
        print('Parameter set {}:\n'.format(parameter_set_index))
        print(values)



if __name__ == "__main__":
    cost_function_names = ndop.optimization.min_values.COST_FUNCTION_NAMES[-7:-3] + ndop.optimization.min_values.COST_FUNCTION_NAMES[-2:]
    print_all_values_for_min_values(cost_function_names)   

