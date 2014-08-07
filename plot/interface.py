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

def optimization_cost_function_for_kind(path='/tmp', data_kind='WOD', y_max=None):
    if data_kind == 'WOD':
        kind_of_cost_functions = ('WOD_OLS_1', 'WOD_WLS_1', 'WOD_GLS_1')
        line_labels = ('WOD-OLS', 'WOD-WLS', 'WOD-GLS')
    else:
        kind_of_cost_functions = ('WOA_OLS_1', 'WOA_WLS_1')
        line_labels = ('WOA-OLS', 'WOA-WLS')
    
    
    file = os.path.join(path, 'optimization_{}_cost_function.png'.format(data_kind))
    line_colors = ('b', 'r', 'g')
    
    n = len(kind_of_cost_functions)
    
    xs = []
    ys = []
    
    ## plot each function call
    for cf_kind in kind_of_cost_functions:
        f = ndop.optimization.results.get_values(cf_kind, 'all_f')
        y = f / f[0]
        x = np.arange(len(y))
        xs.append(x)
        ys.append(y)
    line_labels = list(line_labels)
    line_styles = ['-'] * n
    line_colors = list(line_colors)[:n]
    
    ## plot each iteration step
    for cf_kind in kind_of_cost_functions:
        f = ndop.optimization.results.get_values(cf_kind, 'solver_f')
        y = f / f[0]
        x = ndop.optimization.results.get_values(cf_kind, 'solver_evals_f')
        x -= 1
        xs.append(x)
        ys.append(y)
    line_labels += [None] * n
    line_styles += ['o']*n
    line_colors *= 2
    
    x_label = 'number of function evaluations'
    y_label = '(normalized) function value'
    
    util.plot.line(xs, ys, file, line_label=line_labels, line_style=line_styles, line_color=line_colors, line_width=3, tick_font_size=20, legend_font_size=16, y_max=y_max, x_label=x_label, y_label=y_label, use_log_scale=True)


def optimization_cost_functions(path='/tmp'):
    optimization_cost_function_for_kind(path=path, data_kind='WOD')
    optimization_cost_function_for_kind(path=path, data_kind='WOA')


def optimization_parameters_for_kind(path='/tmp', kind='WOD_OLS_1', all_parameters_in_one_plot=True):
    p_labels = [r'$\lambda}$', r'$\alpha$', r'$\sigma$', r'$K_{phy}$', r'$I_{C}$', r'$K_{w}$', r'$b$']
    p_bounds = np.array([[0.05, 0.95], [0.5, 10], [0.05, 0.95], [0.001, 10], [5, 50], [0.0001, 0.2], [0.7 , 1.3]])
    
    ## get values
    all_ps = ndop.optimization.results.get_values(kind, 'all_p').swapaxes(0,1)
    all_xs = np.arange(all_ps.shape[1])
    
    solver_ps = ndop.optimization.results.get_values(kind, 'solver_p').swapaxes(0,1)
    solver_xs = ndop.optimization.results.get_values(kind, 'solver_evals_f')
    solver_xs -= 1
    
    x_label = 'number of function evaluations'
    n = len(all_ps)
    
    ## plot all normalized parameters in one plot
    if all_parameters_in_one_plot:
        
        ## prepare y values
        def normalize(values):
            ## normalize values to range [0, 1]
            p_lb = p_bounds[:, 0][:, np.newaxis] 
            p_ub = p_bounds[:, 1][:, np.newaxis] 
            values = (values - p_lb) / (p_ub - p_lb)
            return values
        
        all_ps = normalize(all_ps)
        solver_ps = normalize(solver_ps)
        ys = all_ps.tolist() + solver_ps.tolist()
        
        ## prepare other
        xs = [all_xs] * n + [solver_xs] * n
        line_labels = p_labels + [None] * n
        line_styles = ['-'] * n + [ 'o'] * n
        line_colors = util.plot.get_colors(n) * 2
        
        file = os.path.join(path, 'optimization_{}_normalized_parameters.png'.format(kind))
        y_label = 'normalized value'
        [y_min, y_max] = [0, 1]
        
        ## plot
        util.plot.line(xs, ys, file, line_style=line_styles, line_label=line_labels, line_color=line_colors, line_width=3, tick_font_size=20, legend_font_size=16, x_label=x_label, y_label=y_label, y_min=y_min, y_max=y_max)
    
    ## plot each parameter
    else:
        xs = [all_xs, solver_xs]
        line_styles = ['-', 'o']
        
        for i in range(n):
            ys = [all_ps[i], solver_ps[i]]
            y_label = p_labels[i]
            file = os.path.join(path, 'optimization_{}_parameter_{}.png'.format(kind, i))
            [y_min, y_max] = p_bounds[i]
            
            util.plot.line(xs, ys, file, line_style=line_styles, line_width=3, tick_font_size=20, legend_font_size=16, x_label=x_label, y_label=y_label, y_min=y_min, y_max=y_max)


def optimization_parameters(path='/tmp'):
    for kind in ['WOA_OLS_1', 'WOA_WLS_1', 'WOD_OLS_1', 'WOD_WLS_1', 'WOD_GLS_1']:
        optimization_parameters_for_kind(path=path, kind=kind)



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
    from ndop.model.constants import (METOS_X_DIM as X_DIM, METOS_Y_DIM as Y_DIM, METOS_Z_LEFT as Z_VALUES_LEFT)
    
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
            m.transform_indices_to_boxes(X_DIM, Y_DIM, Z_VALUES_LEFT)
            data = measurements.util.map.insert_values_in_map(m.means(), no_data_value=np.inf)
#             if layer is not None:
#                 data = data[:, :, :, layer]
#                 data = data.reshape(data.shape + (1,))
            util.plot.data(data, file, no_data_value=np.inf, vmin=0, vmax=y_max)
        
        plot_tracer_diff(data_base.points[0], data_base.diff_dop(p), file.format(data_kind, 'dop'), y_max=y_max[0])
        plot_tracer_diff(data_base.points[1], data_base.diff_po4(p), file.format(data_kind, 'po4'), y_max=y_max[1])
    else:
        raise ValueError('Data_kind {} unknown. Must be "WOA" or "WOD".'.format(data_kind))
    
    