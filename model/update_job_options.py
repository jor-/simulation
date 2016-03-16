import os
import stat

import ndop.model.job

import util.io.fs
import util.options
import util.logging
logger = util.logging.logger


## general update functions

def update_job_options(update_function):
    from ndop.model.constants import MODEL_NAMES, DATABASE_OUTPUT_DIR, DATABASE_MODEL_DIRNAME, DATABASE_SPINUP_DIRNAME

    for model_name in MODEL_NAMES:
        model_dirname = DATABASE_MODEL_DIRNAME.format(model_name)
        model_dir = os.path.join(DATABASE_OUTPUT_DIR, model_dirname)
        
        for time_step_dir in util.io.fs.get_dirs(model_dir):
            parameter_set_dirs = util.io.fs.get_dirs(time_step_dir)
            logger.debug('{} parameter set dirs found in {}.'.format(len(parameter_set_dirs), time_step_dir))
    
            for parameter_set_dir in parameter_set_dirs:
                spinup_dir = os.path.join(parameter_set_dir, DATABASE_SPINUP_DIRNAME)
                update_job_options_in_run_dirs(spinup_dir, update_function)
    
                derivative_dir = os.path.join(parameter_set_dir, 'derivative')
                
                for step_size_dir in util.io.fs.get_dirs(derivative_dir):
                    for partial_derivative_dir in util.io.fs.get_dirs(step_size_dir):
                        update_job_options_in_run_dirs(partial_derivative_dir, update_function)
    


def update_job_options_in_run_dirs(run_dir_path, update_function):
    from ndop.model.constants import DATABASE_RUN_DIRNAME

    runs_len = len(util.io.fs.get_dirs(run_dir_path))

    for run in range(runs_len):
        run_dirname = DATABASE_RUN_DIRNAME.format(run)
        run_dir = os.path.join(run_dir_path, run_dirname)

        if os.path.exists(run_dir):
            update_job_options_in_job_options_dir(run_dir, update_function)


def update_job_options_in_job_options_dir(job_options_dir, update_function):
    logger.debug('Updating job options in {}.'.format(job_options_dir))

    options_file = os.path.join(job_options_dir, 'job_options.hdf5')

    util.io.fs.make_writable(options_file)
    update_function(job_options_dir)
    util.io.fs.make_read_only(options_file)


## specific update functions

def update_output_dir():
    def update_function(job_options_dir):
        with ndop.model.job.Metos3D_Job(job_options_dir, force_load=True) as job:
            job.update_output_dir(job_options_dir)

    update_job_options(update_function)


def add_finished_file():
    def update_function(job_options_dir):
        options_file = os.path.join(job_options_dir)
        with util.options.Options(options_file, mode='r') as options:
            try:
                options['/job/finished_file']
                print('Finished file option already there in job option file {}.'.format(options_file))
            except KeyError:
                finished_file = os.path.join(job_options_dir, 'finished.txt')
                options['/job/finished_file'] = finished_file
                print('Finished file option added to job option file {}.'.format(options_file))

    update_job_options(update_function)



def update_str_options(old_str, new_str):
    def update_function(job_options_dir):
        options_file = os.path.join(job_options_dir, 'job_options.hdf5')
        with util.options.Options(options_file) as options:
            options.replace_all_str_options(old_str, new_str)
    update_job_options(update_function)


def update_new_option_entries():
    def update_function(job_options_dir):
        options_file = os.path.join(job_options_dir)
        with util.options.Options(options_file, mode='r') as options:
            try:
                options['/model/concentrations']
                print('Concentrations option already there in job option file {}.'.format(options_file))
            except KeyError:
                options['/model/concentrations'] = np.array([2.17, 10**-4])
                print('Concentrations option added to job option file {}.'.format(options_file))
            try:
                options['/model/time_step_multiplier']
                print('time_step_multiplier option already there in job option file {}.'.format(options_file))
            except KeyError:
                options['/model/time_step_multiplier'] = 1
                print('time_step_multiplier option added to job option file {}.'.format(options_file))
            try:
                options['/model/time_steps_per_year']
                print('time_steps_per_year option already there in job option file {}.'.format(options_file))
            except KeyError:
                options['/model/time_steps_per_year'] = options['/model/time_step_count']
                del options['/model/time_step_count']
                print('time_steps_per_year option added to job option file {}.'.format(options_file))

    update_job_options(update_function)


if __name__ == "__main__":
    with util.logging.Logger():
        update_str_options('${NDOP_DIR}/model_output', '${SIMULATION_OUTPUT_DIR}/model_dop_po4')
        update_new_option_entries()
    print('Update completed.')