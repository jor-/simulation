import numpy as np
import os.path

import measurements.util.data
import measurements.util.map
import ndop.optimization.results
import ndop.util.data_base

import util.plot

import logging
logger = logging.getLogger(__name__)


## optimization results

def optimization_results_for_kind(path='/tmp', kind='WOD', y_max=None):
    if kind == 'WOD':
        kind_of_cost_functions = ('WOD_OLS_1', 'WOD_WLS_1', 'WOD_GLS_1')
        line_labels = ('WOD-OLS', 'WOD-WLS', 'WOD-GLS')
    else:
        kind_of_cost_functions = ('WOA_OLS_1', 'WOA_WLS_1')
        line_labels = ('WOA-OLS', 'WOA-WLS')
    
    
    file = os.path.join(path, 'optimization_results_' + kind + '.png')
    line_colors = ('b', 'r', 'g')
    line_width = 3
    
    n = len(kind_of_cost_functions)
    
    xs = []
    ys = []
    
    for cf_kind in kind_of_cost_functions:
        f = ndop.optimization.results.get_values(cf_kind, 'all_f')
        y = f / f[0]
        y = y[1:]
        x = np.arange(len(y))
        xs.append(x)
        ys.append(y)
    line_labels = list(line_labels)
    line_styles = ['-'] * n
    line_colors = list(line_colors)[:n]
    
    for cf_kind in kind_of_cost_functions:
        f = ndop.optimization.results.get_values(cf_kind, 'solver_f')
        y = f / f[0]
        y = y[2:]
        x = ndop.optimization.results.get_values(cf_kind, 'solver_evals_f')
        x -= 1
        x = x[2:]
        xs.append(x)
        ys.append(y)
    line_labels += [None] * n
    line_styles += ['o']*n
    line_colors *= 2
    
    x_label = 'number of function evaluations'
    y_label = '(normalized) function value'
    
    util.plot.line(xs, ys, file, line_label=line_labels, line_style=line_styles, line_color=line_colors, line_width=line_width, tick_font_size=20, legend_font_size=16, y_max=y_max, x_label=x_label, y_label=y_label)


def optimization_results(path='/tmp'):
    optimization_results_for_kind(path=path, kind='WOD', y_max=2)
    optimization_results_for_kind(path=path, kind='WOA', y_max=30)




## model output


def model_output(path='/tmp', parameter_set_nr=0, y_max=(None, None)):
    from ndop.model.constants import MODEL_OUTPUT_DIR, MODEL_TIME_STEP_DIRNAME, MODEL_PARAMETERS_SET_DIRNAME
#     , MODEL_PARAMETERS_FILENAME
    from ndop.util.constants import CACHE_DIRNAME, F_ALL_CACHE_FILENAME
    
    logger.debug('Plotting model output for parameter set {}'.format(parameter_set_nr))
    
    parameter_set_dirname = MODEL_PARAMETERS_SET_DIRNAME.format(parameter_set_nr)
    f_file = os.path.join(MODEL_OUTPUT_DIR, MODEL_TIME_STEP_DIRNAME.format(1), parameter_set_dirname, CACHE_DIRNAME, F_ALL_CACHE_FILENAME)
    f = np.load(f_file)
    
    file = os.path.join(path, 'model_output_' + parameter_set_dirname + '_{}.png')
    util.plot.data(f[0], file.format('dop'), land_value=np.nan, no_data_value=None, vmin=0, vmax=y_max[0])
    util.plot.data(f[1], file.format('po4'), land_value=np.nan, no_data_value=None, vmin=0, vmax=y_max[1])


def model_confidence(path='/tmp', parameter_set_nr=0, kind='WOA_WLS', vmax=(None, None)):
    from ndop.model.constants import MODEL_OUTPUT_DIR, MODEL_TIME_STEP_DIRNAME, MODEL_PARAMETERS_SET_DIRNAME
#     , MODEL_PARAMETERS_FILENAME
    from ndop.accuracy.constants import CACHE_DIRNAME, MODEL_CONFIDENCE_FILENAME
    
    logger.debug('Plotting model confidence')
    
    parameter_set_dirname = MODEL_PARAMETERS_SET_DIRNAME.format(parameter_set_nr)
    f_file = os.path.join(MODEL_OUTPUT_DIR, MODEL_TIME_STEP_DIRNAME.format(1), parameter_set_dirname, CACHE_DIRNAME, kind, MODEL_CONFIDENCE_FILENAME)
    f = np.load(f_file)
    
    file = os.path.join(path, 'model_confidence_' + parameter_set_dirname + '_{}_{}.png')
    util.plot.data(f[0], file.format(kind, 'dop'), land_value=np.nan, no_data_value=None, vmin=0, vmax=vmax[0])
    util.plot.data(f[1], file.format(kind, 'po4'), land_value=np.nan, no_data_value=None, vmin=0, vmax=vmax[1])


def model_diff(path='/tmp', parameter_set_nr=0, data_kind='WOA', y_max=(None, None)):
    from ndop.model.constants import MODEL_OUTPUT_DIR, MODEL_TIME_STEP_DIRNAME, MODEL_PARAMETERS_SET_DIRNAME, MODEL_PARAMETERS_FILENAME
    from ndop.model.constants import (METOS_X_DIM as X_DIM, METOS_Y_DIM as Y_DIM, METOS_Z as Z_VALUES)
    
    logger.debug('Plotting model output for parameter set {}'.format(parameter_set_nr))
    
    ## load parameters
    parameter_set_dirname = MODEL_PARAMETERS_SET_DIRNAME.format(parameter_set_nr)
    p_file = os.path.join(MODEL_OUTPUT_DIR, MODEL_TIME_STEP_DIRNAME.format(1), parameter_set_dirname, MODEL_PARAMETERS_FILENAME)
    p = np.loadtxt(p_file)
    
    ## init data base
    data_base = ndop.util.data_base.init_data_base(data_kind)
    file = os.path.join(path, 'model_diff_' + parameter_set_dirname + '_{}_{}.png')
    
    ## print for WOA
    if data_kind.upper() == 'WOA':
#         diff_all_dop = np.abs(data_base.diff_all_dop(p))
#         diff_all_po4 = np.abs(data_base.diff_all_po4(p))
#         
#         file = os.path.join(path, parameter_set_dirname + '_model_diff_dop.png')
#         util.plot.data(diff_all_dop, file, land_value=np.nan, no_data_value=np.inf, vmin=0, vmax=y_max[0])
#         file = os.path.join(path, parameter_set_dirname + '_model_diff_po4.png')
#         util.plot.data(diff_all_po4, file, land_value=np.nan, no_data_value=np.inf, vmin=0, vmax=y_max[1])
        
        def plot_tracer_diff(diff, file, y_max=None):
            util.plot.data(np.abs(diff), file, land_value=np.nan, no_data_value=np.inf, vmin=0, vmax=y_max)
        
        plot_tracer_diff(data_base.diff_all_dop(p), file.format(data_kind, 'dop'), y_max=y_max[0])
        plot_tracer_diff(data_base.diff_all_po4(p), file.format(data_kind, 'po4'), y_max=y_max[1])
    
    ## print for WOD
    elif data_kind.upper() == 'WOD':
        def plot_tracer_diff(points, diff, file, y_max=None):
            m = measurements.util.data.Measurements_Unsorted()
            m.add_results(points, np.abs(diff))
            m.discard_year()
            m.categorize_indices([1./12])
            m.transform_indices_to_boxes(X_DIM, Y_DIM, Z_VALUES)
            data = measurements.util.map.insert_values_in_map(m.means(), no_data_value=np.inf)
#             if layer is not None:
#                 data = data[:, :, :, layer]
#                 data = data.reshape(data.shape + (1,))
            util.plot.data(data, file, no_data_value=np.inf, vmin=0, vmax=y_max)
        
        plot_tracer_diff(data_base.points[0], data_base.diff_dop(p), file.format(data_kind, 'dop'), y_max=y_max[0])
        plot_tracer_diff(data_base.points[1], data_base.diff_po4(p), file.format(data_kind, 'po4'), y_max=y_max[1])
    else:
        raise ValueError('Data_kind {} unknown. Must be "WOA" or "WOD".'.format(data_kind))
    
    