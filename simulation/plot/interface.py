import numpy as np
import os
import os.path

import simulation.optimization.results
import simulation.optimization.min_values
import simulation.util.data_base
import simulation.constants

import util.plot
import util.logging
logger = util.logging.logger


def get_kind(data_kind, setup_index=3):
    data_kinds = [cost_function_name for cost_function_name in simulation.optimization.min_values.COST_FUNCTION_NAMES if cost_function_name.startswith(data_kind) and not 'LWLS' in cost_function_name]
    data_kinds_with_setup = ['setup_{}/'.format(setup_index) + data_kind for data_kind in data_kinds]
    if 'OLD' in data_kind:
        data_kinds_with_setup = [dk for dk in data_kinds_with_setup if not '30' in dk]
    return data_kinds_with_setup

def get_label(kind):
    if 'OLD' in kind:
        if 'OLS' in kind:
            return 'OLS'
        if 'WLS' in kind:
            return 'WLS'
        if '25' in kind:
            return 'GLS-40'
        if '35' in kind:
            return 'GLS-45'
        if '40' in kind:
            return 'GLS-50'
    kind = kind.replace('min_values_', '').replace('max_year_diff_', '').replace('inf', '').replace('//', '/')
    if kind.endswith('/'):
        kind = kind[:-1]
    kind = kind.replace('/', '-')
    return kind


## optimization results

def optimization_cost_function_for_data_kind(data_kind='WOD', path='/tmp', y_max=None, with_line_search_steps=True, number_of_function_evals_max=-1):

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
    
    ## get data
    def get_data(kind_of_cost_functions):
        f_all = simulation.optimization.results.all_f(kind_of_cost_functions)
        
        if number_of_function_evals_max > 0 and len(f_all) > number_of_function_evals_max:
            f_all = f_all[:number_of_function_evals_max]
        
        if len(f_all) > 0:
            f_all = f_all / f_all[0]
        
        x_all = np.arange(len(f_all))
        return x_all, f_all
    

    ## plot each function call
    if with_line_search_steps:
        for i in range(n):
            x_all, f_all = get_data(kind_of_cost_functions[i])
            if len(f_all) > 0:
                xs.append(x_all)
                ys.append(f_all)
                line_labels.append(None)
                line_styles.append('*')
                line_colors.append(line_colors_all[i])
                line_widths.append(2)

    ## plot each iteration step
    for i in range(n):
        x_all, f_all = get_data(kind_of_cost_functions[i])
        if len(f_all) > 0:
            local_solver_runs_list = simulation.optimization.results.local_solver_runs_list(kind_of_cost_functions[i])
            for j in range(len(local_solver_runs_list)):
                local_solver_run = local_solver_runs_list[j]
                if number_of_function_evals_max > 0:
                    local_solver_run = local_solver_run[local_solver_run < number_of_function_evals_max]
                xs.append(x_all[local_solver_run])
                ys.append(f_all[local_solver_run])
                line_styles.append('-')
                line_colors.append(line_colors_all[i])
                line_widths.append(3)
                if j == 0:
                    line_label = get_label(kind_of_cost_functions[i])
                    line_labels.append(line_label)
                else:
                    line_labels.append(None)

    util.plot.line(xs, ys, file, line_label=line_labels, line_style=line_styles, line_color=line_colors, line_width=line_widths, tick_font_size=20, legend_font_size=16, y_max=y_max, use_log_scale=True)


def optimization_cost_functions(path='/tmp', y_max=10, with_line_search_steps=True):
    for data_kind in ('WOA', 'WOD', 'WOD_TMM_1', 'WOD_TMM_0'):
        optimization_cost_function_for_data_kind(data_kind=data_kind, path=path, y_max=y_max, with_line_search_steps=with_line_search_steps)


def optimization_parameters_for_kind(kind, path='/tmp', all_parameters_in_one_plot=True, with_line_search_steps=True, number_of_function_evals_max=-1):
    p_labels = [r'$\lambda}$', r'$\alpha$', r'$\sigma$', r'$K_{phy}$', r'$I_{C}$', r'$K_{w}$', r'$b$']
    kind_label = get_label(kind)

    ## get values
    all_p = simulation.optimization.results.all_p(kind)
    if number_of_function_evals_max > 0 and len(all_p) > number_of_function_evals_max:
        all_p = all_p[:number_of_function_evals_max]
    all_p = all_p.swapaxes(0,1)

    if len(all_p) > 0:
        all_x = np.arange(all_p.shape[1])

        n = len(all_p)

        p_bounds =  np.ones([2, n])
        p_bounds[0] = - p_bounds[0]

        ## plot all normalized parameters in one plot
        if all_parameters_in_one_plot:

            ## prepare parameter values
            p_lb = p_bounds[0][:, np.newaxis]
            p_ub = p_bounds[1][:, np.newaxis]

            def normalize(values):
                ## normalize values to range [0, 1]
                values = (values - p_lb) / (p_ub - p_lb)
                return values

            all_p = normalize(all_p)

            ## prepare plot all values
            if with_line_search_steps:
                xs = [all_x]*n
                ys = all_p.tolist()
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

            ## prepare plot local solver line
            local_solver_runs_list = simulation.optimization.results.local_solver_runs_list(kind)
            for j in range(len(local_solver_runs_list)):
                local_solver_run = local_solver_runs_list[j]
                if number_of_function_evals_max > 0:
                    local_solver_run = local_solver_run[local_solver_run < number_of_function_evals_max]
                xs.extend([all_x[local_solver_run]]*n)
                ys.extend(all_p.T[local_solver_run].T.tolist())
                line_styles.extend(['-']*n)
                line_colors.extend(util.plot.get_colors(n))
                line_widths.extend([3]*n)
                if j == 0:
                    line_labels.extend(p_labels)
                else:
                    line_labels.extend([None]*n)


            ## prepare rest
            file = os.path.join(path, 'optimization_normalized_parameters_-_{}.png'.format(kind_label))
            [y_min, y_max] = [0, 1]

            ## plot
            util.plot.line(xs, ys, file, line_style=line_styles, line_label=line_labels, line_color=line_colors, line_width=line_widths, tick_font_size=20, legend_font_size=16, y_min=y_min, y_max=y_max)

        ## plot each parameter
        else:
            solver_x = simulation.optimization.results.solver_f_indices(kind)
            solver_p = simulation.optimization.results.solver_p(kind).swapaxes(0,1)
            xs = [all_x, solver_x]
            line_styles = ['-', 'o']
            for i in range(n):
                ys = [all_p[i], solver_p[i]]
                file = os.path.join(path, 'optimization_{}_parameter_{}.png'.format(kind_label, i))
                [y_min, y_max] = p_bounds[:, i]

                util.plot.line(xs, ys, file, line_style=line_styles, line_width=3, tick_font_size=20, legend_font_size=16, y_min=y_min, y_max=y_max)



def optimization_parameters_for_data_kind(data_kind, path='/tmp', all_parameters_in_one_plot=True, with_line_search_steps=True):
    for kind in get_kind(data_kind):
        optimization_parameters_for_kind(path=path, kind=kind, with_line_search_steps=with_line_search_steps)
    

def optimization_parameters(path='/tmp', with_line_search_steps=True):
    for kind in simulation.optimization.min_values.COST_FUNCTION_NAMES:
        optimization_parameters_for_kind(path=path, kind=kind, with_line_search_steps=with_line_search_steps)


def optimization(path='/tmp', with_line_search_steps=True):
    optimization_cost_functions(path=path, with_line_search_steps=with_line_search_steps)
    optimization_parameters(path=path, with_line_search_steps=with_line_search_steps)



## model output


def model_output(parameter_set_nr, kind='BOXES', path='/tmp', y_max=(None, None), average_in_time=False):
    from simulation.model.constants import DATABASE_OUTPUT_DIR, DATABASE_MODEL_DIRNAME, DATABASE_TIME_STEP_DIRNAME, DATABASE_PARAMETERS_DIRNAME, DATABASE_PARAMETERS_FILENAME
    from simulation.util.constants import CACHE_DIRNAME, BOXES_F_FILENAME, WOD_F_FILENAME

    logger.debug('Plotting model output for parameter set {}'.format(parameter_set_nr))

    ## load parameters
    parameter_set_dirname = DATABASE_PARAMETERS_DIRNAME.format(parameter_set_nr)
    p_file = os.path.join(DATABASE_OUTPUT_DIR, DATABASE_MODEL_DIRNAME.format('dop_po4'), DATABASE_TIME_STEP_DIRNAME.format(1), parameter_set_dirname, DATABASE_PARAMETERS_FILENAME)
    p = np.loadtxt(p_file)

    ## init data base
    if kind.upper() == 'BOXES':
        data_base = simulation.util.data_base.init_data_base('WOA')
        f = data_base.f_boxes(p)
        f[f < 0] = 0
        if average_in_time:
            f = f.mean(axis=1)
    else:
        data_base = simulation.util.data_base.init_data_base('WOD')
        f = data_base.f(p)
        f = data_base.convert_to_boxes(f, no_data_value=np.inf)

    file = os.path.join(path, 'model_output_-_' + kind + '_-_' + parameter_set_dirname + '_-_{tracer}.png')
    tracers = ('dop', 'po4')
    for i in range(len(tracers)):
        util.plot.data(f[i], file.format(tracer=tracers[i]), land_value=np.nan, no_data_value=np.inf, v_min=0, v_max=y_max[i], contours=True, colorbar=False)



def relative_parameter_confidence(parameter_set_nr, kind='WOA_WLS', path='/tmp'):
    from simulation.model.constants import DATABASE_OUTPUT_DIR, DATABASE_MODEL_DIRNAME, DATABASE_TIME_STEP_DIRNAME, DATABASE_PARAMETERS_DIRNAME, DATABASE_PARAMETERS_FILENAME
    from simulation.accuracy.constants import CACHE_DIRNAME, PARAMETER_CONFIDENCE_FILENAME

    logger.debug('Plotting parameter confidence for parameter set {}'.format(parameter_set_nr))

    ## load value
    parameter_dirname = DATABASE_PARAMETERS_DIRNAME.format(parameter_set_nr)
    parameter_dir = os.path.join(DATABASE_OUTPUT_DIR, DATABASE_MODEL_DIRNAME.format('dop_po4'), DATABASE_TIME_STEP_DIRNAME.format(1), parameter_dirname)
    parameter_value_file = os.path.join(parameter_dir, DATABASE_PARAMETERS_FILENAME)
    parameter_value = np.loadtxt(parameter_value_file)
    parameter_confidence_file = os.path.join(parameter_dir, CACHE_DIRNAME, kind, PARAMETER_CONFIDENCE_FILENAME)
    parameter_confidence = np.load(parameter_confidence_file)
    
    relative_parameter_confidence_percent = 100 * parameter_confidence / parameter_value
    relative_parameter_confidence_percent_interval = [relative_parameter_confidence_percent, - relative_parameter_confidence_percent]
    
    ## plot
    file = os.path.join(path, 'relative_parameter_confidence_-_' + parameter_dirname + '_-_' + get_label(kind) + '.png')
    util.plot.intervals(relative_parameter_confidence_percent_interval, file, use_percent_ticks=True)



def model_confidence(parameter_set_nr, kind='WOA_WLS', path='/tmp', v_max=[None, None], time_dim_confidence=12, time_dim_df=12, average_in_time=False):
    from simulation.model.constants import DATABASE_OUTPUT_DIR, DATABASE_MODEL_DIRNAME, DATABASE_TIME_STEP_DIRNAME, DATABASE_PARAMETERS_DIRNAME
    from simulation.accuracy.constants import CACHE_DIRNAME, MODEL_CONFIDENCE_FILENAME

    logger.debug('Plotting model confidence for parameter set {}'.format(parameter_set_nr))

    ## load value
    parameter_set_dirname = DATABASE_PARAMETERS_DIRNAME.format(parameter_set_nr)
    f_file = os.path.join(DATABASE_OUTPUT_DIR, DATABASE_MODEL_DIRNAME.format('dop_po4'), DATABASE_TIME_STEP_DIRNAME.format(1), parameter_set_dirname, CACHE_DIRNAME, kind, MODEL_CONFIDENCE_FILENAME.format(time_dim_confidence=time_dim_confidence, time_dim_df=time_dim_df))
    f = np.load(f_file)
    if average_in_time:
        f = f.mean(axis=1)
    assert len(f) == 2
    
    ## set v_max
    for i in range(len(f)):
        if v_max[i] is None:
            v_max[i] = np.nanmax(f[i])
            rounding_exponent = np.sign(np.log10(v_max[i])) * np.ceil(np.abs(np.log10(v_max[i])))
            v_max[i] = np.floor(v_max[i] * 10**(-rounding_exponent)) * 10**rounding_exponent
    
    ## plot
    file = os.path.join(path, 'model_confidence_-_' + parameter_set_dirname + '_-_' + get_label(kind) + '_-_time_dim_df_{}'.format(time_dim_df) + '_-_{tracer}.png')
    tracers = ('dop', 'po4')
    for i in range(len(tracers)):
        util.plot.data(f[i], file.format(tracer=tracers[i]), land_value=np.nan, no_data_value=None, v_min=0, v_max=v_max[i], contours=True, colorbar=False)


def average_model_confidence_increase(parameter_set_nr, kind='WOA_WLS', path='/tmp', time_dim_df=12):
    from simulation.model.constants import DATABASE_OUTPUT_DIR, DATABASE_MODEL_DIRNAME, DATABASE_TIME_STEP_DIRNAME, DATABASE_PARAMETERS_DIRNAME
    from simulation.accuracy.constants import CACHE_DIRNAME, AVERAGE_MODEL_CONFIDENCE_INCREASE_FILENAME

    logger.debug('Plotting average model confidence increase for parameter set {}'.format(parameter_set_nr))

    parameter_set_dirname = DATABASE_PARAMETERS_DIRNAME.format(parameter_set_nr)

    f_file = os.path.join(DATABASE_OUTPUT_DIR, DATABASE_MODEL_DIRNAME.format('dop_po4'), DATABASE_TIME_STEP_DIRNAME.format(1), parameter_set_dirname, CACHE_DIRNAME, kind, AVERAGE_MODEL_CONFIDENCE_INCREASE_FILENAME.format(time_dim_df))
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


def model_diff(parameter_set_nr, data_kind='WOA', path='/tmp', normalize_with_deviation=False, y_max=(None, None)):
    from simulation.model.constants import DATABASE_OUTPUT_DIR, DATABASE_MODEL_DIRNAME, DATABASE_TIME_STEP_DIRNAME, DATABASE_PARAMETERS_DIRNAME, DATABASE_PARAMETERS_FILENAME
    from simulation.model.constants import (METOS_X_DIM as X_DIM, METOS_Y_DIM as Y_DIM, METOS_Z_LEFT as Z_VALUES_LEFT)

    logger.debug('Plotting model output for parameter set {}'.format(parameter_set_nr))

    ## load parameters
    parameter_set_dirname = DATABASE_PARAMETERS_DIRNAME.format(parameter_set_nr)
    p_file = os.path.join(DATABASE_OUTPUT_DIR, DATABASE_MODEL_DIRNAME.format('dop_po4'), DATABASE_TIME_STEP_DIRNAME.format(1), parameter_set_dirname, DATABASE_PARAMETERS_FILENAME)
    p = np.loadtxt(p_file)

    ## init data base
    data_base = simulation.util.data_base.init_data_base(data_kind)
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
