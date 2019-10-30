import pathlib
import os.path

import numpy as np

import util.logging
import util.plot.save

import measurements.plot.data

import simulation.model.cache
import simulation.model.constants
import simulation.plot.util


def model_output(model_options, time_dim, tracer=None, plot_type='all', v_max=None, overwrite=False, colorbar=True, **kwargs):
    model = simulation.model.cache.Model(model_options=model_options)
    model_lsm = model.model_lsm
    if tracer is None:
        tracers = model_options.tracers
    else:
        tracers = (tracer,)
    for tracer in tracers:
        f_all = model.f_all(time_dim, tracers=(tracer,))[tracer]
        base_file = simulation.plot.util.filename(model, f'model_output_-_time_dim_{time_dim}', tracer)
        measurements.plot.data.plot(f_all, base_file, model_lsm, plot_type=plot_type, v_max=v_max, overwrite=overwrite, colorbar=colorbar, **kwargs)


def _filename_with_accuracy_object(accuracy_object, plot_kind, plot_name):
    model = accuracy_object.model
    measurements_name = str(accuracy_object.measurements)
    accuracy_name = f'{accuracy_object.__class__.__name__}({accuracy_object.measurements.correlation_id})'
    file = simulation.plot.util.filename(model, os.path.join(plot_kind, measurements_name, accuracy_name), plot_name)
    return file


def parameters_confidences(accuracy_object, matrix_type='F_H', alpha=0.99, include_variance_factor=True,
                           use_interval_length=True, relative=True, **kwargs):
    plot_kind = 'parameters_confidences'
    plot_name = f'parameters_confidences_-_interval_length_{use_interval_length}_-_matrix_type_{matrix_type}_-_include_variance_factor_{include_variance_factor}_-_alpha_{alpha}_-_relative_{relative}'
    file = _filename_with_accuracy_object(accuracy_object, plot_kind, plot_name)
    data = accuracy_object.parameter_confidence(matrix_type=matrix_type, alpha=alpha, include_variance_factor=include_variance_factor, relative=relative)
    model_name = accuracy_object.model.model_options.model_name
    parameters_names = simulation.model.constants.MODEL_PARAMETER_NAMES[model_name]
    util.logging.debug(f'Plotting parameter confidences at {file}')
    if use_interval_length:
        data *= 2
        tick_transform_y = lambda tick: f'${tick:.0%}$'.replace('%', '\\%')
    else:
        tick_transform_y = lambda tick: f'$\\pm {tick:.1%}$'.replace('%', '\\%')
    util.plot.save.bar(file, data, x_labels=parameters_names, tick_transform_y=tick_transform_y, **kwargs)


def parameters_correlations(accuracy_object, matrix_type='F_H', **kwargs):
    plot_kind = 'parameters_correlations'
    plot_name = f'parameters_correlations_-_matrix_type_{matrix_type}'
    file = _filename_with_accuracy_object(accuracy_object, plot_kind, plot_name)
    correlation_matrix = accuracy_object.correlation_matrix(matrix_type=matrix_type)
    model_name = accuracy_object.model.model_options.model_name
    parameters_names = simulation.model.constants.MODEL_PARAMETER_NAMES[model_name]
    util.logging.debug(f'Plotting parameter correlations at {file}')
    util.plot.save.dense_matrix_pattern(file, correlation_matrix, colorbar=True, x_tick_lables=parameters_names, y_tick_lables=parameters_names, **kwargs)


def model_confidences(accuracy_object, matrix_type='F_H', alpha=0.99, include_variance_factor=True, use_interval_length=True,
                      time_dim_model=12, time_dim_confidence=12,
                      tracer=None, plot_type='all', v_max=None, overwrite=False, colorbar=True, **kwargs):
    # check tracer
    tracers = accuracy_object.model.model_options.tracers
    if tracer is not None and tracer not in tracers:
        raise ValueError(f'Tracer {tracer} is unkown. Only the tracers {tracers} are in the model.')

    # get data
    model_lsm = accuracy_object.model.model_lsm
    data = accuracy_object.model_confidence(matrix_type=matrix_type, alpha=alpha, include_variance_factor=include_variance_factor, time_dim_model=time_dim_model, time_dim_confidence=time_dim_confidence)
    assert len(data) == len(tracers)
    if use_interval_length:
        data *= 2

    # transform ticks
    tick_transform_exponent = util.plot.auxiliary.tick_transform_function_exponent_notation(data, v_max=v_max)

    if use_interval_length:
        tick_transform = tick_transform_exponent
    else:
        def tick_transform(tick):
            tick = tick_transform_exponent(tick)
            tick = '$\\pm' + tick[1:]
            return tick

    if plot_type.startswith('depth'):
        tick_transform_dict = {'tick_transform_x': tick_transform}
    elif plot_type in ('time', 'histogram'):
        tick_transform_dict = {'tick_transform_y': tick_transform}
    else:
        # tick_transform_dict = {'colorbar_tick_transform': tick_transform}
        tick_transform_dict = {}

    # plot
    plot_kind = 'model_confidences'
    for i, tracer_i in enumerate(tracers):
        if tracer is None or tracer_i == tracer:
            plot_name = f'model_confidences_-_interval_length_{use_interval_length}_-_{tracer_i}_-_matrix_type_{matrix_type}_-_include_variance_factor_{include_variance_factor}_-_alpha_{alpha}_-_time_dim_model_{time_dim_model}_-_time_dim_confidence_{time_dim_confidence}'
            base_file = _filename_with_accuracy_object(accuracy_object, plot_kind, plot_name)
            util.logging.debug(f'Plotting model confidences at {base_file}')
            measurements.plot.data.plot(data[i], base_file, model_lsm, plot_type=plot_type, v_max=v_max, overwrite=overwrite, colorbar=colorbar, **tick_transform_dict, **kwargs)


def confidence_increases(accuracy_object, confidence_type='average_model_confidence', number_of_measurements=1,
                         alpha=0.99, include_variance_factor=True, use_interval_length=True,
                         increases_calculation_relative=True, confidence_increases_relative_to_confidence=True,
                         time_dim_model=12, time_dim_confidence=12,
                         tracer=None, plot_type='all', v_max=None, overwrite=False, colorbar=True, **kwargs):
    # check tracer
    tracers = accuracy_object.model.model_options.tracers
    if tracer is not None and tracer not in tracers:
        raise ValueError(f'Tracer {tracer} is unkown. Only the tracers {tracers} are in the model.')

    # get data
    model_lsm = accuracy_object.model.model_lsm
    data = accuracy_object.confidence_increase(
        confidence_type=confidence_type, number_of_measurements=number_of_measurements, alpha=alpha, include_variance_factor=include_variance_factor,
        increases_calculation_relative=increases_calculation_relative, confidence_increases_relative_to_confidence=confidence_increases_relative_to_confidence,
        time_dim_model=time_dim_model, time_dim_confidence=time_dim_confidence)
    assert len(data) == len(tracers)

    if use_interval_length:
        data *= 2

    # transform ticks
    tick_transform = util.plot.auxiliary.tick_transform_function_exponent_notation(data, v_max=v_max)

    if plot_type.startswith('depth'):
        tick_transform_dict = {'tick_transform_x': tick_transform}
    elif plot_type in ('time', 'histogram'):
        tick_transform_dict = {'tick_transform_y': tick_transform}
    else:
        tick_transform_dict = {'colorbar_tick_transform': tick_transform}

    # plot
    plot_kind = f'confidence_increases_-_{confidence_type}'
    for i, tracer_i in enumerate(tracers):
        if tracer is None or tracer_i == tracer:
            plot_name = f'increases_-_length_{use_interval_length}_-_{tracer_i}_-_measurements_{number_of_measurements}_-_relative_confidence_{increases_calculation_relative}_-_relative_increases_{confidence_increases_relative_to_confidence}_-_variance_factor_{include_variance_factor}_-_alpha_{alpha}_-_dim_model_{time_dim_model}_-_dim_increase_{time_dim_confidence}'
            base_file = _filename_with_accuracy_object(accuracy_object, plot_kind, plot_name)
            util.logging.debug(f'Plotting model confidences increases at {base_file}')
            measurements.plot.data.plot(data[i], base_file, model_lsm, plot_type=plot_type, v_max=v_max, overwrite=overwrite, colorbar=colorbar, **tick_transform_dict, **kwargs)
