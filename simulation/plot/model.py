import pathlib

import numpy as np

import util.plot.auxiliary
import util.plot.save

import simulation.model.cache


def output(model_options, tracer=None, time_dim=1, path=None, y_max=None, contours=True, colorbar=False):
    model = simulation.model.cache.Model(model_options=model_options)
    if tracer is not None:
        tracers = (tracer,)
    else:
        tracers = model_options.tracers
    if path is not None:
        path = pathlib.PurePath(path)

    for tracer in tracers:
        f_all = model.f_all(time_dim, tracers=(tracer,))[tracer]
        v_min = 0
        if y_max is None:
            v_max = util.plot.auxiliary.v_max(f_all)
        else:
            v_max = y_max
        filename = f'model_output_{model_options.model_name}_-_concentrations_index_{model.initial_concentration_dir_index}_-_parameters_index_{model.parameter_set_dir_index}_-_{tracer}_-_time_dim_{time_dim}_-_contours_{contours}_-_colorbar_{colorbar}_-_y_max_{y_max}.png'
        plot_file = pathlib.PurePath(filename)
        if path is not None:
            plot_file = path.joinpath(plot_file)
        util.plot.save.data(plot_file, f_all, land_value=np.nan, no_data_value=np.inf, v_min=v_min, v_max=v_max, contours=contours, colorbar=colorbar)
