import numpy as np
import os.path

import measurements.util.data
import ndop.optimization.results
import ndop.util.data_base

import util.plot

import logging
logger = logging.getLogger(__name__)


## optimization results

def optimization_cost_function_for_kind(path='/tmp', data_kind='WOD', y_max=None):
    
    ## init
    if data_kind == 'WOD':
        kind_of_cost_functions = ('WOD_OLS_1', 'WOD_WLS_1', 'WOD_GLS_1', 'WOD_LWLS_1')
        line_labels_all = ('WOD-OLS', 'WOD-WLS', 'WOD-GLS', 'WOD-LWLS')
    elif data_kind == 'WOA':
        kind_of_cost_functions = ('WOA_OLS_1', 'WOA_WLS_1', 'WOA-LWLS_1')
        line_labels_all = ('WOA-OLS', 'WOA-WLS', 'WOA_LWLS')
    elif data_kind == 'OLD_WOD':
        kind_of_cost_functions = ('OLD_WOD_WLS_1', 'OLD_WOD_LWLS_1')
        line_labels_all = ('OLD_WOD_WLS', 'OLD_WOD_LWLS')
    else:
        raise ValueError('Unknown data kind {}.'.format(data_kind))
    
    
    file = os.path.join(path, 'optimization_cost_function_-_{}.png'.format(data_kind))
    line_colors_all = ('b', 'r', 'g', 'k')
    
    n = len(kind_of_cost_functions)
    
    xs = []
    ys = []
    line_labels = []
    line_styles = []
    line_colors = []
    
    ## plot each function call
    for i in range(n):
        f = ndop.optimization.results.get_values(kind_of_cost_functions[i], 'all_f')
        if len(f) > 0:
            y = f / f[0]
            x = np.arange(len(y))
            
            xs.append(x)
            ys.append(y)
            line_labels.append(line_labels_all[i])
            line_styles.append('--')
            line_colors.append(line_colors_all[i])
    
    ## plot each iteration step
    for i in range(n):
        f = ndop.optimization.results.get_values(kind_of_cost_functions[i], 'solver_f')
        if len(f) > 0:
            y = f / f[0]
            x = ndop.optimization.results.get_values(kind_of_cost_functions[i], 'solver_eval_f_index')
            
            xs.append(x)
            ys.append(y)
            line_labels.append(None)
            line_styles.append('o')
            line_colors.append(line_colors_all[i])
            
            xs.append(x)
            ys.append(y)
            line_labels.append(None)
            line_styles.append('-')
            line_colors.append(line_colors_all[i])
    
    x_label = 'number of function evaluations'
    y_label = '(normalized) function value'
    
    util.plot.line(xs, ys, file, line_label=line_labels, line_style=line_styles, line_color=line_colors, line_width=3, tick_font_size=20, legend_font_size=16, y_max=y_max, x_label=x_label, y_label=y_label, use_log_scale=True)


def optimization_cost_functions(path='/tmp', y_max=10):
    optimization_cost_function_for_kind(path=path, data_kind='WOD', y_max=y_max)
    optimization_cost_function_for_kind(path=path, data_kind='WOA', y_max=y_max)
    optimization_cost_function_for_kind(path=path, data_kind='OLD_WOD', y_max=y_max)


def optimization_parameters_for_kind(path='/tmp', kind='WOD_OLS_1', all_parameters_in_one_plot=True):
    from ndop.optimization.constants import PARAMETER_BOUNDS
    p_labels = [r'$\lambda}$', r'$\alpha$', r'$\sigma$', r'$K_{phy}$', r'$I_{C}$', r'$K_{w}$', r'$b$']
    
#     p_bounds = np.array([[0.05, 0.95], [0.5, 10], [0.05, 0.95], [0.005, 10], [10, 50], [0.001, 0.2], [0.7 , 1.3]])
    
    ## get values
    all_ps = ndop.optimization.results.get_values(kind, 'all_p').swapaxes(0,1)
    
    if len(all_ps) > 0:
        all_xs = np.arange(all_ps.shape[1])
        
        solver_ps = ndop.optimization.results.get_values(kind, 'solver_p').swapaxes(0,1)
        solver_xs = ndop.optimization.results.get_values(kind, 'solver_eval_f_index')
        
        x_label = 'number of function evaluations'
        n = len(all_ps)
        
        ## plot all normalized parameters in one plot
        if all_parameters_in_one_plot:
            
            ## prepare y values
            def normalize(values):
                ## normalize values to range [0, 1]
#                 p_lb = p_bounds[:, 0][:, np.newaxis] 
#                 p_ub = p_bounds[:, 1][:, np.newaxis] 
                p_lb = PARAMETER_BOUNDS[0][:, np.newaxis]
                p_ub = PARAMETER_BOUNDS[1][:, np.newaxis]
                values = (values - p_lb) / (p_ub - p_lb)
                return values
            
            all_ps = normalize(all_ps)
            solver_ps = normalize(solver_ps)
            ys = all_ps.tolist() + solver_ps.tolist() * 2
            
            ## prepare other
#             xs = [all_xs] * n + [solver_xs] * n
#             line_labels = p_labels + [None] * n
#             line_styles = ['-'] * n + [ 'o'] * n
#             line_colors = util.plot.get_colors(n) * 2
            xs = [all_xs] * n + [solver_xs] * n * 2
            line_labels = p_labels + [None] * n * 2
            line_styles = ['--'] * n + [ 'o'] * n + ['-'] * n
            line_colors = util.plot.get_colors(n) * 3
            
            file = os.path.join(path, 'optimization_normalized_parameters_-_{}.png'.format(kind))
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
#                 [y_min, y_max] = p_bounds[i]
                [y_min, y_max] = PARAMETER_BOUNDS[:, i]
                
                util.plot.line(xs, ys, file, line_style=line_styles, line_width=3, tick_font_size=20, legend_font_size=16, x_label=x_label, y_label=y_label, y_min=y_min, y_max=y_max)


def optimization_parameters(path='/tmp'):
    for kind in ['WOA_OLS_1', 'WOA_WLS_1', 'WOD_OLS_1', 'WOD_WLS_1', 'WOD_GLS_1', 'WOD_LWLS_1', 'WOA_LWLS_1', 'OLD_WOD_LWLS_1', 'OLD_WOD_WLS_1']:
        optimization_parameters_for_kind(path=path, kind=kind)
        

def optimization(path='/tmp'):
    optimization_cost_functions(path=path)
    optimization_parameters(path=path)



## model output


def model_output(path='/tmp', parameter_set_nr=0, kind='BOXES', y_max=(None, None)):
    from ndop.model.constants import MODEL_OUTPUT_DIR, MODEL_TIME_STEP_DIRNAME, MODEL_PARAMETERS_SET_DIRNAME, MODEL_PARAMETERS_FILENAME
    from ndop.util.constants import CACHE_DIRNAME, F_BOXES_CACHE_FILENAME, F_WOD_CACHE_FILENAME
    
    logger.debug('Plotting model output for parameter set {}'.format(parameter_set_nr))
    
    ## load parameters
    parameter_set_dirname = MODEL_PARAMETERS_SET_DIRNAME.format(parameter_set_nr)
    p_file = os.path.join(MODEL_OUTPUT_DIR, MODEL_TIME_STEP_DIRNAME.format(1), parameter_set_dirname, MODEL_PARAMETERS_FILENAME)
    p = np.loadtxt(p_file)
    
    ## init data base
    if kind.upper() == 'BOXES':
        data_base = ndop.util.data_base.init_data_base('WOA')
        f = data_base.f_boxes(p)
    else:
        data_base = ndop.util.data_base.init_data_base('WOD')
        f = data_base.F(p)
        f = data_base.convert_to_boxes(f, no_data_value=np.inf)
    
    
    
#     parameter_set_dirname = MODEL_PARAMETERS_SET_DIRNAME.format(parameter_set_nr)
#     f_dir = os.path.join(MODEL_OUTPUT_DIR, MODEL_TIME_STEP_DIRNAME.format(1), parameter_set_dirname, CACHE_DIRNAME)
#     if kind.upper() == 'BOXES':
#         f_file = os.path.join(f_dir, F_BOXES_CACHE_FILENAME.format(12))
#         f = np.load(f_file)
#     else:
#         f_file = os.path.join(f_dir, F_WOD_CACHE_FILENAME)
#         f = np.load(f_file)
        
    plot_file = os.path.join(path, 'model_output_-_' + kind + '_-_' + parameter_set_dirname + '_-_{}.png')
    
    util.plot.data(f[0], plot_file.format('dop'), land_value=np.nan, no_data_value=np.inf, vmin=0, vmax=y_max[0])
    util.plot.data(f[1], plot_file.format('po4'), land_value=np.nan, no_data_value=np.inf, vmin=0, vmax=y_max[1])


def model_confidence(path='/tmp', parameter_set_nr=0, kind='WOA_WLS', vmax=(None, None), time_dim_df=12):
    from ndop.model.constants import MODEL_OUTPUT_DIR, MODEL_TIME_STEP_DIRNAME, MODEL_PARAMETERS_SET_DIRNAME
    from ndop.accuracy.constants import CACHE_DIRNAME, MODEL_CONFIDENCE_FILENAME
    
    logger.debug('Plotting model confidence for parameter set {}'.format(parameter_set_nr))
    
    parameter_set_dirname = MODEL_PARAMETERS_SET_DIRNAME.format(parameter_set_nr)
    
    f_file = os.path.join(MODEL_OUTPUT_DIR, MODEL_TIME_STEP_DIRNAME.format(1), parameter_set_dirname, CACHE_DIRNAME, kind, MODEL_CONFIDENCE_FILENAME.format(tim_dim_confidence=12, time_dim_df=time_dim_df))
    f = np.load(f_file)
    
    file = os.path.join(path, 'model_confidence_-_' + parameter_set_dirname + '_-_time_dim_df_{}'.format(time_dim_df) + '_-_{}_-_{}.png')
    util.plot.data(f[0], file.format(kind, 'dop'), land_value=np.nan, no_data_value=None, vmin=0, vmax=vmax[0])
    util.plot.data(f[1], file.format(kind, 'po4'), land_value=np.nan, no_data_value=None, vmin=0, vmax=vmax[1])


def average_model_confidence_increase(path='/tmp', parameter_set_nr=0, kind='WOA_WLS', time_dim_df=12):
    from ndop.model.constants import MODEL_OUTPUT_DIR, MODEL_TIME_STEP_DIRNAME, MODEL_PARAMETERS_SET_DIRNAME
    from ndop.accuracy.constants import CACHE_DIRNAME, AVERAGE_MODEL_CONFIDENCE_INCREASE_FILENAME
    
    logger.debug('Plotting average model confidence increase for parameter set {}'.format(parameter_set_nr))
    
    parameter_set_dirname = MODEL_PARAMETERS_SET_DIRNAME.format(parameter_set_nr)
    
    f_file = os.path.join(MODEL_OUTPUT_DIR, MODEL_TIME_STEP_DIRNAME.format(1), parameter_set_dirname, CACHE_DIRNAME, kind, AVERAGE_MODEL_CONFIDENCE_INCREASE_FILENAME.format(time_dim_df))
    f = np.load(f_file)
    
    vmin = np.nanmin(f, axis=tuple(np.arange(f.ndim-1)+1))
    vmax = np.nanmax(f, axis=tuple(np.arange(f.ndim-1)+1))
    significant_digits=3
    for i in range(len(vmin)):
        round_factor = 10 ** (np.ceil(-np.log10(vmin[i])) + significant_digits)
        vmin[i] = np.floor(vmin[i] * round_factor) / round_factor
    for i in range(len(vmax)):
        round_factor = 10 ** (np.ceil(-np.log10(vmin[i])) + significant_digits)
        vmax[i] = np.ceil(vmax[i] * round_factor) / round_factor
    
    file = os.path.join(path, 'average_model_confidence_increase_-_' + parameter_set_dirname + '_-_time_dim_df_{}'.format(time_dim_df) + '_-_{}_-_{}.png')
    util.plot.data(f[0], file.format(kind, 'dop'), land_value=np.nan, no_data_value=None, vmin=vmin[0], vmax=vmax[0])
    util.plot.data(f[1], file.format(kind, 'po4'), land_value=np.nan, no_data_value=None, vmin=vmin[1], vmax=vmax[1])


def model_diff(path='/tmp', parameter_set_nr=0, data_kind='WOA', normalize_with_deviation=False, y_max=(None, None)):
    from ndop.model.constants import MODEL_OUTPUT_DIR, MODEL_TIME_STEP_DIRNAME, MODEL_PARAMETERS_SET_DIRNAME, MODEL_PARAMETERS_FILENAME
    from ndop.model.constants import (METOS_X_DIM as X_DIM, METOS_Y_DIM as Y_DIM, METOS_Z_LEFT as Z_VALUES_LEFT)
    
    logger.debug('Plotting model output for parameter set {}'.format(parameter_set_nr))
    
    ## load parameters
    parameter_set_dirname = MODEL_PARAMETERS_SET_DIRNAME.format(parameter_set_nr)
    p_file = os.path.join(MODEL_OUTPUT_DIR, MODEL_TIME_STEP_DIRNAME.format(1), parameter_set_dirname, MODEL_PARAMETERS_FILENAME)
    p = np.loadtxt(p_file)
    
    ## init data base
    data_base = ndop.util.data_base.init_data_base(data_kind)
    if not normalize_with_deviation:
        file = os.path.join(path, 'model_diff_-_{}_-_' + parameter_set_dirname + '_-_{}.png')
    else:
        file = os.path.join(path, 'model_diff_normalized_with_deviation_-_{}_-_' + parameter_set_dirname + '_-_{}.png')
    
    ## print for WOA
    if data_kind.upper() == 'WOA':
        diff_boxes = np.abs(data_base.diff_boxes(p, normalize_with_deviation=normalize_with_deviation))
    ## print for WOD
    elif data_kind.upper() == 'WOD':
        diff = np.abs(data_base.diff(p, normalize_with_deviation=normalize_with_deviation))
        diff_boxes = data_base.convert_to_boxes(diff, no_data_value=np.inf)
    else:
        raise ValueError('Data_kind {} unknown. Must be "WOA" or "WOD".'.format(data_kind))
    
    
    def plot_tracer_diff(diff, file, y_max=None):
        util.plot.data(np.abs(diff), file, land_value=np.nan, no_data_value=np.inf, vmin=0, vmax=y_max)
    
    plot_tracer_diff(diff_boxes[0], file.format(data_kind, 'dop'), y_max=y_max[0])
    plot_tracer_diff(diff_boxes[1], file.format(data_kind, 'po4'), y_max=y_max[1])
    