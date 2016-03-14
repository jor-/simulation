import sys
import argparse

import os
import numpy as np

import util.pattern
import util.io.fs

import ndop.model.eval
from ndop.optimization.matlab.constants import DATA_KINDS, GLS_DICT

DATA_KINDS = [data_kind for data_kind in DATA_KINDS if not data_kind.startswith('OLD')]
COST_FUNCTION_NAMES = ['{data_kind}/{cost_function}'.format(data_kind=dk.replace('.', '_TMM_'), cost_function=cf) for dk in DATA_KINDS for cf in ('OLS', 'WLS', 'LWLS')] + ['{data_kind}/GLS/min_values_{min_values}/max_year_diff_inf/min_diag_1e-01'.format(data_kind=dk.replace('.', '_TMM_'), min_values=mv) for dk in DATA_KINDS for mv in GLS_DICT[dk]] + ['OLDWOD_TMM_1/{cost_function}'.format(cost_function=cf) for cf in ('OLS', 'WLS', 'LWLS')] + ['OLDWOD_TMM_1/GLS/min_values_{min_values}/max_year_diff_inf/min_diag_1e-02'.format(min_values=mv) for mv in GLS_DICT['OLDWOD.1']]
COST_FUNCTION_NAMES.sort()

COST_FUNCTION_OUTPUT_DIRNAME = 'cost_functions'
COST_FUNCTION_F_FILENAME = 'f.txt'


def values_dict(parameter_set_index, cost_function_names):
    m = ndop.model.eval.Model()
    parameter_set_dir = m.parameter_set_dir_with_index(parameter_set_index)
    
    values = {}
    for cost_function_name in cost_function_names:
        cost_function_f_file = os.path.join(parameter_set_dir, COST_FUNCTION_OUTPUT_DIRNAME, cost_function_name, COST_FUNCTION_F_FILENAME)
        try:
            values[cost_function_name] = np.loadtxt(cost_function_f_file)
        except FileNotFoundError:
            pass
    return values


def min_values_dicts(cost_function_names):
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
        current_values_dict = values_dict(parameter_set_index, cost_function_names)
        for key, value in current_values_dict.items():
            if value < best_values[key]:
                best_indices[key] = int(parameter_set_index)
                best_values[key] = value
                
    return (best_indices, best_values)


def values_array(parameter_set_index, cost_function_names):
    m = ndop.model.eval.Model()
    parameter_set_dir = m.parameter_set_dir_with_index(parameter_set_index)
    
    n = len(cost_function_names)
    values = np.empty(n)
    for i in range(n):
        cost_function_f_file = os.path.join(parameter_set_dir, COST_FUNCTION_OUTPUT_DIRNAME, cost_function_names[i], COST_FUNCTION_F_FILENAME)
        try:
            values[i] = np.loadtxt(cost_function_f_file)
        except FileNotFoundError:
            values[i] = np.nan
    return values


def min_values_arrays(cost_function_names):
    ## init arrays
    cost_function_names = np.asarray(cost_function_names)
    n = len(cost_function_names)
    best_indices = np.empty(n, dtype=np.int32)
    best_values = np.empty(n)
    for i in range(n):
        best_indices[i] = -1
        best_values[i] = np.inf
    
    ## check alls parameter sets
    m = ndop.model.eval.Model()
    used_indices = m._parameter_db.used_indices()
    for parameter_set_index in used_indices:
        current_values = values_array(parameter_set_index, cost_function_names)
        for i in range(n):
            if current_values[i] < best_values[i]:
                best_indices[i] = int(parameter_set_index)
                best_values[i] = current_values[i]
    
    ## return
    mask = best_indices >= 0
    cost_function_names = cost_function_names[mask]
    best_indices = best_indices[mask]
    best_values = best_values[mask]
    return (cost_function_names, best_indices, best_values)



def all_values_for_min_values(cost_function_names):
    (cost_function_names, best_indices, best_values) = min_values_arrays(cost_function_names)
    n = len(cost_function_names)
    all_values = np.empty(n, n)
    for i in range(n):
        current_values = values_array(best_indices[i], cost_function_names)
        all_values[i] = current_values
    return cost_function_names, all_values


def all_normalized_values_for_min_values(cost_function_names):
    (cost_function_names, best_indices, best_values) = min_values_arrays(cost_function_names)
    n = len(cost_function_names)
    all_normalized_values = np.empty([n, n])
    for i in range(n):
        current_values = values_array(best_indices[i], cost_function_names)
        all_normalized_values[i] = current_values / best_values
    return cost_function_names, all_normalized_values, best_indices, best_values


def print_all_values_for_min_values(cost_function_names):
    cost_function_names, all_normalized_values, best_indices, best_values = all_normalized_values_for_min_values(cost_function_names)
    m = ndop.model.eval.Model()
    
    array_formatter = lambda value: '{:.0%}'.format(value)
    
    def print_for_parameter_set(cost_function_name, index, values):
        def array_to_str(array):
            array_str = np.array2string(array, formatter={'all': array_formatter})
            array_str = array_str.replace('\n', '').replace('\r', '')
            return array_str
        p = m._parameter_db.get_value(index)
        p_str = array_to_str(p)
        print(': Parameter set {:d}: {} (Best for cost function {})'.format(index, p_str, cost_function_name))
        print(array_to_str(values))
        print('')

    cost_function_names = [cf.replace('/max_year_diff_inf/min_diag_1e-02', '').replace('min_values_','') for cf in cost_function_names]
    print('All min cost function values for:')
    print(cost_function_names)
    print_for_parameter_set('', 0, values_array(0, cost_function_names) / best_values)
    for i in range(len(best_indices)):
        if best_indices[i] >= 0:
            print_for_parameter_set(cost_function_names[i], best_indices[i], all_normalized_values[i])



if __name__ == "__main__":
    cost_function_names = [cf for cf in COST_FUNCTION_NAMES if 'WOD_TMM_1' in cf and not 'LWLS' in cf]
    print_all_values_for_min_values(cost_function_names)   

