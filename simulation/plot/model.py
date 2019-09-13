import pathlib

import numpy as np

import util.plot.save

import simulation.model.save
import simulation.model.cache


def output(model_name, concentrations_index, parameters_index, tracer=None, time_dim=1, path=None, y_max=None):
    model_options = simulation.model.save.prepare_model_options(model_name, time_step=1, concentrations_index=concentrations_index, parameter_set_index=parameters_index)
    model = simulation.model.cache.Model(model_options=model_options)
    if tracer is not None:
        tracers = (tracer,)
    else:
        tracers = model_options.tracers
    if path is not None:
        path = pathlib.PurePath(path)

    for tracer in tracers:
        f_all = model.f_all(time_dim, tracers=(tracer,))
        filename = f'model_output_-_{model_name}_-_concentrations_index_{concentrations_index}_-_parameters_index_{parameters_index}_-_time_dim_{time_dim}_-_{tracer}.png'
        plot_file = pathlib.PurePath(filename)
        if path is not None:
            plot_file = path.joinpath(plot_file)
        util.plot.save.data(plot_file, f_all, land_value=np.nan, no_data_value=np.inf, v_min=0, v_max=y_max[i], contours=True, colorbar=False)
