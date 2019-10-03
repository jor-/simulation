import pathlib

import numpy as np

import measurements.plot.data

import simulation.model.cache
import simulation.plot.util


def output(model_options, time_dim, plot_type='all', v_max=None, overwrite=False, colorbar=True, **kwargs):
    model = simulation.model.cache.Model(model_options=model_options)
    model_lsm = model.model_lsm
    for tracer in model_options.tracers:
        f_all = model.f_all(time_dim, tracers=(tracer,))[tracer]
        base_file = simulation.plot.util.filename(model, 'model_output', tracer)
        measurements.plot.data.plot(f_all, base_file, model_lsm, plot_type=plot_type, v_max=v_max, overwrite=overwrite, colorbar=colorbar, **kwargs)
