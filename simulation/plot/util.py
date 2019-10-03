import pathlib
import warnings

import simulation.plot.constants


def filename(model_object, kind, plot_name):
    file = simulation.plot.constants.PLOT_FILE_WITHOUT_FILE_EXTENSION.format(
        model_name=model_object.model_options.model_name,
        time_step=model_object.model_options.time_step,
        concentrations_index=model_object.initial_concentration_dir_index,
        parameters_index=model_object.parameter_set_dir_index,
        kind=kind,
        plot_name=plot_name)
    # check for bad chars
    bad_char = '.'
    replacement_char = '_'
    if bad_char in file:
        warnings.warn(f'A "{bad_char}" is in the filename {file} which should be avoided. Replacing "{bad_char}" by "{replacement_char}".')
        file = file.replace(bad_char, replacement_char)
    assert bad_char not in file
    # append file extension
    file_extension = simulation.plot.constants.PLOT_DEFAULT_FILE_EXTENSION
    assert not file_extension.startswith('.')
    file += '.' + file_extension
    path = pathlib.PurePath(file)
    # return
    return file
