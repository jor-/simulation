import numpy as np
import os
import os.path

import measurements.util.data
import ndop.optimization.results
import ndop.optimization.min_values
import ndop.util.data_base
import ndop.constants

import util.plot
import util.logging
logger = util.logging.logger


def get_kind(data_kind):
    data_kind = data_kind.upper()+'/'
    return [cfn for cfn in ndop.optimization.min_values.COST_FUNCTION_NAMES if cfn.startswith(data_kind)]

def get_label(kind):
    return kind.replace('/', '-').replace('min_values_', '').replace('max_year_diff_', '').replace('inf', '')


## optimization results

def optimization_cost_function_for_kind(data_kind='WOD', path='/tmp', y_max=None, with_line_search_steps=True):

    ## init
    kind_of_cost_functions = get_kind(data_kind)

    file = os.path.join(path, 'optimization_cost_function_-_{}.png'.format(data_kind))
    n = len(kind_of_cost_functions)
    line_colors_all = util.plot.get_colors(n)

    xs = []
    ys = []
    line_labels = []
    line_styles = []
    line_colors = []
    line_widths = []

    ## plot each function call
    if with_line_search_steps:
        for i in range(n):
            f = ndop.optimization.results.all_f(kind_of_cost_functions[i])
            if len(f) > 0:
                y = f / f[0]
                x = np.arange(len(y))

                xs.append(x)
                ys.append(y)
                line_labels.append(None)
                line_styles.append('*')
                line_colors.append(line_colors_all[i])
                line_widths.append(2)

    ## plot each iteration step
    for i in range(n):
        kind_of_cost_function = kind_of_cost_functions[i]
        f = ndop.optimization.results.solver_f(kind_of_cost_function)
        if len(f) > 0:
            y = f / f[0]
            x = ndop.optimization.results.solver_f_indices(kind_of_cost_function)
            local_stop_slices = ndop.optimization.results.local_solver_stop_slices(kind_of_cost_function)

            for j in range(len(local_stop_slices)):
                slice = local_stop_slices[j]
                xs.append(x[slice])
                ys.append(y[slice])
                line_styles.append('-')
                line_colors.append(line_colors_all[i])
                line_widths.append(3)
                if j == 0:
                    # line_label = LABELS[kind_of_cost_function]
                    line_label = get_label(kind_of_cost_function)
                    line_labels.append(line_label)
                else:
                    line_labels.append(None)


    x_label = 'number of function evaluations'
    y_label = '(normalized) function value'

    util.plot.line(xs, ys, file, line_label=line_labels, line_style=line_styles, line_color=line_colors, line_width=line_widths, tick_font_size=20, legend_font_size=16, y_max=y_max, x_label=x_label, y_label=y_label, use_log_scale=True)


def optimization_cost_functions(path='/tmp', y_max=10, with_line_search_steps=True):
    for data_kind in ('WOA', 'WOD', 'WOD_TMM_1', 'WOD_TMM_0'):
        optimization_cost_function_for_kind(data_kind=data_kind, path=path, y_max=y_max, with_line_search_steps=with_line_search_steps)


def optimization_parameters_for_kind(kind, path='/tmp', all_parameters_in_one_plot=True, with_line_search_steps=True):
    # from ndop.optimization.constants import PARAMETER_BOUNDS
    p_labels = [r'$\lambda}$', r'$\alpha$', r'$\sigma$', r'$K_{phy}$', r'$I_{C}$', r'$K_{w}$', r'$b$']
    # kind_label = LABELS[kind]
    kind_label = get_label(kind)

#     p_bounds = np.array([[0.05, 0.95], [0.5, 10], [0.05, 0.95], [0.005, 10], [10, 50], [0.001, 0.2], [0.7 , 1.3]])

    ## get values
    all_ps = ndop.optimization.results.all_p(kind).swapaxes(0,1)

    if len(all_ps) > 0:
        all_xs = np.arange(all_ps.shape[1])

        solver_ps = ndop.optimization.results.solver_p(kind).swapaxes(0,1)
        solver_xs = ndop.optimization.results.solver_f_indices(kind)

        x_label = 'number of function evaluations'
        n = len(all_ps)

        p_bounds =  ndop.optimization.results.p_bounds(kind)

        ## plot all normalized parameters in one plot
        if all_parameters_in_one_plot:

            ## prepare parameter values
            p_lb = p_bounds[0][:, np.newaxis]
            p_ub = p_bounds[1][:, np.newaxis]

            def normalize(values):
                ## normalize values to range [0, 1]
                values = (values - p_lb) / (p_ub - p_lb)
                return values

            all_ps = normalize(all_ps)
            solver_ps = normalize(solver_ps)
#             ys = all_ps.tolist() + solver_ps.tolist() * 2
#
#             ## prepare other
#             xs = [all_xs] * n + [solver_xs] * n * 2
#             line_labels = p_labels + [None] * n * 2
#             line_styles = [':'] * n + ['o'] * n + ['-'] * n
#             line_colors = util.plot.get_colors(n) * 3

            ## prepare plot all values
            if with_line_search_steps:
                xs = [all_xs]*n
                ys = all_ps.tolist()
                line_labels = [None]*n
                line_styles = ['.']*n
                line_colors = util.plot.get_colors(n)
                line_widths = [2]*n
            else:
                xs = []
                ys = []
                line_labels = []
                line_styles = []
                line_colors = []
                line_widths = []

#             xs.extend([solver_xs]*n*2)
#             ys.extend(solver_ps.tolist()*2)
#             line_labels.extend(p_labels)
#             line_labels.extend([None]*n)
#             line_styles.extend(['-']*n)
#             line_styles.extend(['o']*n)
#             line_colors.extend(util.plot.get_colors(n)*2)

            ## prepare plot local solver line
            local_stop_slices = ndop.optimization.results.local_solver_stop_slices(kind)
            for j in range(len(local_stop_slices)):
                slice = local_stop_slices[j]
                xs.extend([solver_xs[slice]]*n)
                ys.extend(solver_ps.T[slice].T.tolist())
                line_styles.extend(['-']*n)
                line_colors.extend(util.plot.get_colors(n))
                line_widths.extend([3]*n)
                if j == 0:
                    line_labels.extend(p_labels)
                else:
                    line_labels.extend([None]*n)

#             ## prepare plot o marker
#             xs.extend([solver_xs]*n)
#             ys.extend(solver_ps.tolist())
#             line_labels.extend([None]*n)
#             line_styles.extend(['o']*n)
#             line_colors.extend(util.plot.get_colors(n))


            ## prepare rest
            file = os.path.join(path, 'optimization_normalized_parameters_-_{}.png'.format(kind_label))
            y_label = 'normalized value'
            [y_min, y_max] = [0, 1]

            ## plot
            util.plot.line(xs, ys, file, line_style=line_styles, line_label=line_labels, line_color=line_colors, line_width=line_widths, tick_font_size=20, legend_font_size=16, x_label=x_label, y_label=y_label, y_min=y_min, y_max=y_max)

        ## plot each parameter
        else:
            xs = [all_xs, solver_xs]
            line_styles = ['-', 'o']

            for i in range(n):
                ys = [all_ps[i], solver_ps[i]]
                y_label = p_labels[i]
                file = os.path.join(path, 'optimization_{}_parameter_{}.png'.format(kind_label, i))
#                 [y_min, y_max] = p_bounds[i]
                [y_min, y_max] = p_bounds[:, i]

                util.plot.line(xs, ys, file, line_style=line_styles, line_width=3, tick_font_size=20, legend_font_size=16, x_label=x_label, y_label=y_label, y_min=y_min, y_max=y_max)


def optimization_parameters(path='/tmp', with_line_search_steps=True):
    for kind in ndop.optimization.min_values.COST_FUNCTION_NAMES:
        optimization_parameters_for_kind(path=path, kind=kind, with_line_search_steps=with_line_search_steps)


def optimization(path='/tmp', with_line_search_steps=True):
    optimization_cost_functions(path=path, with_line_search_steps=with_line_search_steps)
    optimization_parameters(path=path, with_line_search_steps=with_line_search_steps)



## model output


def model_output(path='/tmp', parameter_set_nr=0, kind='BOXES', y_max=(None, None)):
    from ndop.model.constants import MODEL_OUTPUT_DIR, MODEL_TIME_STEP_DIRNAME, MODEL_PARAMETERS_SET_DIRNAME, MODEL_PARAMETERS_FILENAME
    from ndop.util.constants import CACHE_DIRNAME, BOXES_F_FILENAME, WOD_F_FILENAME

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
#         f_file = os.path.join(f_dir, BOXES_F_FILENAME.format(12))
#         f = np.load(f_file)
#     else:
#         f_file = os.path.join(f_dir, WOD_F_FILENAME)
#         f = np.load(f_file)

    plot_file = os.path.join(path, 'model_output_-_' + kind + '_-_' + parameter_set_dirname + '_-_{}.png')

    util.plot.data(f[0], plot_file.format('dop'), land_value=np.nan, no_data_value=np.inf, v_min=0, v_max=y_max[0])
    util.plot.data(f[1], plot_file.format('po4'), land_value=np.nan, no_data_value=np.inf, v_min=0, v_max=y_max[1])


def model_confidence(path='/tmp', parameter_set_nr=0, kind='WOA_WLS', v_max=(None, None), time_dim_df=12):
    from ndop.model.constants import MODEL_OUTPUT_DIR, MODEL_TIME_STEP_DIRNAME, MODEL_PARAMETERS_SET_DIRNAME
    from ndop.accuracy.constants import CACHE_DIRNAME, MODEL_CONFIDENCE_FILENAME

    logger.debug('Plotting model confidence for parameter set {}'.format(parameter_set_nr))

    parameter_set_dirname = MODEL_PARAMETERS_SET_DIRNAME.format(parameter_set_nr)

    f_file = os.path.join(MODEL_OUTPUT_DIR, MODEL_TIME_STEP_DIRNAME.format(1), parameter_set_dirname, CACHE_DIRNAME, kind, MODEL_CONFIDENCE_FILENAME.format(tim_dim_confidence=12, time_dim_df=time_dim_df))
    f = np.load(f_file)

    file = os.path.join(path, 'model_confidence_-_' + parameter_set_dirname + '_-_time_dim_df_{}'.format(time_dim_df) + '_-_{}_-_{}.png')
    util.plot.data(f[0], file.format(kind, 'dop'), land_value=np.nan, no_data_value=None, v_min=0, v_max=v_max[0])
    util.plot.data(f[1], file.format(kind, 'po4'), land_value=np.nan, no_data_value=None, v_min=0, v_max=v_max[1])


def average_model_confidence_increase(path='/tmp', parameter_set_nr=0, kind='WOA_WLS', time_dim_df=12):
    from ndop.model.constants import MODEL_OUTPUT_DIR, MODEL_TIME_STEP_DIRNAME, MODEL_PARAMETERS_SET_DIRNAME
    from ndop.accuracy.constants import CACHE_DIRNAME, AVERAGE_MODEL_CONFIDENCE_INCREASE_FILENAME

    logger.debug('Plotting average model confidence increase for parameter set {}'.format(parameter_set_nr))

    parameter_set_dirname = MODEL_PARAMETERS_SET_DIRNAME.format(parameter_set_nr)

    f_file = os.path.join(MODEL_OUTPUT_DIR, MODEL_TIME_STEP_DIRNAME.format(1), parameter_set_dirname, CACHE_DIRNAME, kind, AVERAGE_MODEL_CONFIDENCE_INCREASE_FILENAME.format(time_dim_df))
    f = np.load(f_file)

    v_min = np.nanmin(f, axis=tuple(np.arange(f.ndim-1)+1))
    v_max = np.nanmax(f, axis=tuple(np.arange(f.ndim-1)+1))
    significant_digits=3
    for i in range(len(v_min)):
        round_factor = 10 ** (np.ceil(-np.log10(v_min[i])) + significant_digits)
        v_min[i] = np.floor(v_min[i] * round_factor) / round_factor
    for i in range(len(v_max)):
        round_factor = 10 ** (np.ceil(-np.log10(v_min[i])) + significant_digits)
        v_max[i] = np.ceil(v_max[i] * round_factor) / round_factor

    file = os.path.join(path, 'average_model_confidence_increase_-_' + parameter_set_dirname + '_-_time_dim_df_{}'.format(time_dim_df) + '_-_{}_-_{}.png')
    util.plot.data(f[0], file.format(kind, 'dop'), land_value=np.nan, no_data_value=None, v_min=v_min[0], v_max=v_max[0])
    util.plot.data(f[1], file.format(kind, 'po4'), land_value=np.nan, no_data_value=None, v_min=v_min[1], v_max=v_max[1])


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
        util.plot.data(np.abs(diff), file, land_value=np.nan, no_data_value=np.inf, v_min=0, v_max=y_max)

    plot_tracer_diff(diff_boxes[0], file.format(data_kind, 'dop'), y_max=y_max[0])
    plot_tracer_diff(diff_boxes[1], file.format(data_kind, 'po4'), y_max=y_max[1])
