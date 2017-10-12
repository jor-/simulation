import os

import numpy as np

import simulation.constants
import simulation.model.job
import simulation.model.constants

import util.io.fs
import util.options
import util.logging


# general update functions for job options

def update_job_options(update_function, model_names=None):
    if model_names is None:
        database_dir = simulation.model.constants.DATABASE_OUTPUT_DIR
        util.logging.info('Getting jobs in {}.'.format(database_dir))
        job_files = util.io.fs.get_files(database_dir, filename_pattern='*/job_options.hdf5', use_absolute_filenames=True, recursive=True)
        util.logging.info('Got {} jobs.'.format(len(job_files)))
    else:
        job_files = []
        for model_name in model_names:
            model_dirname = simulation.model.constants.DATABASE_MODEL_DIRNAME.format(model_name)
            model_dir = os.path.join(simulation.model.constants.DATABASE_OUTPUT_DIR, model_dirname)
            util.logging.info('Getting jobs in {}.'.format(model_dir))
            model_job_files = util.io.fs.get_files(model_dir, filename_pattern='*/job_options.hdf5', use_absolute_filenames=True, recursive=True)
            util.logging.info('Got {} jobs.'.format(len(model_job_files)))
            job_files.extend(model_job_files)

    for job_file in job_files:
        util.io.fs.make_writable(job_file)
        update_function(job_file)
        util.io.fs.make_read_only(job_file)


# specific update functions for job options

def rename_option(job_file, old_name, new_name):
    with util.options.OptionsFile(job_file) as options:
        try:
            value = options[old_name]
        except KeyError:
            try:
                value = options[new_name]
            except KeyError:
                util.logging.error('Option {} is not found in {}.'.format(old_name, new_name, job_file))
            else:
                util.logging.debug('Option {} is already renamed to {} in {}.'.format(old_name, new_name, job_file))
        else:
            options[new_name] = value
            del options[old_name]
            util.logging.info('Option {} renamed to {} in {}.'.format(old_name, new_name, job_file))


def update_function_output_dir(job_file):
    with util.options.OptionsFile(job_file) as options:
        old_output_dir = options['/metos3d/output_dir']
        if old_output_dir.endswith('/'):
            old_output_dir = old_output_dir[:-1]

        new_output_dir = os.path.dirname(job_file)
        new_output_dir = new_output_dir.replace(simulation.constants.SIMULATION_OUTPUT_DIR, '${{{}}}'.format((simulation.constants.SIMULATION_OUTPUT_DIR_ENV_NAME)))

        if new_output_dir != old_output_dir:
            util.logging.info('Changing output path from {} to {}.'.format(old_output_dir, new_output_dir))
            options.replace_all_str_options(old_output_dir, new_output_dir)


def update_tracer_names(job_file, old_tracer_name, new_tracer_name):
    with util.options.OptionsFile(job_file) as options:
        old_tracers = options['/model/tracers']
        if old_tracer_name in old_tracers:
            util.logging.info('Tracer {} found in {}, replacing with {}.'.format(old_tracer_name, job_file, new_tracer_name))

            # replace in model tracers
            new_tracers = [tracer
                           if tracer != old_tracer_name
                           else new_tracer_name
                           for tracer in old_tracers]
            options['/model/tracers'] = new_tracers

            # replace in tracers output filenames
            old_tracer_output_filenames = options['/metos3d/tracer_output_filenames']
            old_tracer_output_filename = '{}_output.petsc'.format(old_tracer_name)
            new_tracer_output_filename = '{}_output.petsc'.format(new_tracer_name)
            new_tracer_output_filenames = [tracer_output_filename
                                           if tracer_output_filename != old_tracer_output_filename
                                           else new_tracer_output_filename
                                           for tracer_output_filename in old_tracer_output_filenames]
            options['/metos3d/tracer_output_filenames'] = new_tracer_output_filenames

            # replace in tracers input filenames
            try:
                old_tracer_input_filenames = options['/metos3d/tracer_input_filenames']
            except KeyError:
                pass
            else:
                old_tracer_input_filename = '{}_input.petsc'.format(old_tracer_name)
                new_tracer_input_filename = '{}_input.petsc'.format(new_tracer_name)
                new_tracer_input_filenames = [tracer_input_filename
                                              if tracer_input_filename != old_tracer_input_filename
                                              else new_tracer_input_filename
                                              for tracer_input_filename in old_tracer_input_filenames]
                options['/metos3d/tracer_input_filenames'] = new_tracer_input_filenames
        else:
            util.logging.debug('Tracer {} not found in {}.'.format(old_tracer_name, old_tracer_name))


def update_run_dirs_in_job_options():
    def update_function(job_file):
        with util.options.OptionsFile(job_file) as options:
            for i in range(9):
                old_str = 'run_{:0>2d}/'.format(i)
                new_str = 'run_{:0>5d}/'.format(i)
                options.replace_all_str_options(old_str, new_str)
    update_job_options(update_function)


def update_tracer_input_files_in_job_options():
    def update_function(job_file):
        with util.options.OptionsFile(job_file) as options:
            try:
                options['/model/tracer_input_files']
            except KeyError:
                pass
            else:
                metos_tracer_input_dir = options['/metos3d/tracer_input_dir']

                if 'derivative' in metos_tracer_input_dir:
                    base_dir = metos_tracer_input_dir[:metos_tracer_input_dir.find('derivative')]
                else:
                    assert 'spinup' in metos_tracer_input_dir
                    base_dir = metos_tracer_input_dir[:metos_tracer_input_dir.find('spinup')]

                model_tracer_input_files_old = options['/model/tracer_input_files']

                model_tracer_input_files_new = tuple(base_dir + model_tracer_input_file[model_tracer_input_file.find('spinup'):] for model_tracer_input_file in model_tracer_input_files_old)
                model_tracer_input_files_new = tuple(model_tracer_input_file.replace('input.petsc', 'output.petsc') for model_tracer_input_file in model_tracer_input_files_new)

                options['/model/tracer_input_files'] = model_tracer_input_files_new

                util.logging.info('Changing "/metos3d/tracer_input_dir" from {} to {}.'.format(model_tracer_input_files_old, model_tracer_input_files_new))
    update_job_options(update_function)


# general update functions for parameter files

def update_parameter_files(update_function):
    from simulation.model.constants import MODEL_NAMES, DATABASE_OUTPUT_DIR, DATABASE_MODEL_DIRNAME, DATABASE_PARAMETERS_FILENAME

    for model_name in MODEL_NAMES:
        model_dirname = DATABASE_MODEL_DIRNAME.format(model_name)
        model_dir = os.path.join(DATABASE_OUTPUT_DIR, model_dirname)

        for time_step_dir in util.io.fs.get_dirs(model_dir, use_absolute_filenames=True):
            parameter_set_dirs = util.io.fs.get_dirs(time_step_dir, use_absolute_filenames=True)
            util.logging.debug('{} parameter set dirs found in {}.'.format(len(parameter_set_dirs), time_step_dir))

            for parameter_set_dir in parameter_set_dirs:
                parameter_file = os.path.join(parameter_set_dir, DATABASE_PARAMETERS_FILENAME)

                util.io.fs.make_writable(parameter_file)
                p = np.loadtxt(parameter_file)
                p = update_function(p)
                np.savetxt(parameter_file, p)
                util.io.fs.make_read_only(parameter_file)


# main function

def _main():

    # parse arguments
    import argparse

    parser = argparse.ArgumentParser(description='Update job options in database to current version.')
    parser.add_argument('--debug_level', choices=util.logging.LEVELS, default='INFO', help='Print debug infos low to passed level.')
    parser.add_argument('--version', action='version', version='%(prog)s {}'.format(simulation.__version__))
    args = parser.parse_args()

    # run
    with util.logging.Logger(level=args.debug_level):
        def update_function(job_file):
            rename_option(job_file, '/model/tracer', '/model/tracers')
            update_tracer_names(job_file, 'p', 'phytoplankton')
            update_tracer_names(job_file, 'z', 'zooplankton')
            update_tracer_names(job_file, 'd', 'detritus')
        update_job_options(update_function)

    util.logging.info('Update completed.')


if __name__ == "__main__":
    _main()
