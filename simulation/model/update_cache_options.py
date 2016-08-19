import numpy as np

import simulation.model.constants

import util.io.fs
import util.io.np


def update_cache_derivative_options():
    value_cache_option_files = util.io.fs.filter_with_filename_pattern(simulation.model.constants.DATABASE_OUTPUT_DIR, '*options.npy', exclude_dirs=True, use_absolute_filenames=True, recursive=True)
    for value_cache_option_file in value_cache_option_files:
        value_cache_option = np.load(value_cache_option_file)
        if len(value_cache_option) == 4:
            new_value_cache_option = np.array([value_cache_option[0], value_cache_option[1], value_cache_option[2], simulation.model.constants.MODEL_DEFAULT_DERIVATIVE_OPTIONS['years'], simulation.model.constants.MODEL_DEFAULT_DERIVATIVE_OPTIONS['step_size'], value_cache_option[3]])
            print('Updating value {} to {} in {}'.format(value_cache_option, new_value_cache_option, value_cache_option_file))
            assert len(new_value_cache_option) == 6
            util.io.np.save_npy_and_txt(value_cache_option_file, new_value_cache_option, make_read_only=True, overwrite=True)
        else:
            assert len(value_cache_option) in [3, 6]


if __name__ == "__main__":
    update_cache_derivative_options()
    print('Update completed.')