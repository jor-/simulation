import os
import stat

import ndop.model.job

import util.io.fs
import util.options
import util.logging
logger = util.logging.logger


## general update functions

def update_job_options(update_function):
    from ndop.model.constants import MODEL_OUTPUT_DIR, MODEL_TIME_STEP_DIRNAME, MODEL_PARAMETERS_SET_DIRNAME, MODEL_SPINUP_DIRNAME, MODEL_RUN_DIRNAME, MODEL_DERIVATIVE_DIRNAME, MODEL_PARTIAL_DERIVATIVE_DIRNAME
    
    partial_derivatives = range(7)
    h_factors = (-1, 1)
    
    time_step_sizes=(1,)
    for time_step_size in time_step_sizes:
        time_step_dirname = MODEL_TIME_STEP_DIRNAME.format(time_step_size)
        time_step_dir = os.path.join(MODEL_OUTPUT_DIR, time_step_dirname)
        
        parameter_sets_len = len(util.io.fs.get_dirs(time_step_dir))
        logger.info('{} parameter set dirs found in {}.'.format(parameter_sets_len, time_step_dir))
        
        for parameter_set_number in range(parameter_sets_len):
            parameter_set_dirname = MODEL_PARAMETERS_SET_DIRNAME.format(parameter_set_number)
            parameter_set_dir = os.path.join(time_step_dir, parameter_set_dirname)
            
            spinup_dir = os.path.join(parameter_set_dir, MODEL_SPINUP_DIRNAME)
            
            update_job_options_in_run_dirs(spinup_dir, update_function)
            
            derivative_dir = os.path.join(parameter_set_dir, MODEL_DERIVATIVE_DIRNAME.format(10**(-7)))
            
            for partial_derivative in partial_derivatives:
                for h_factor in h_factors:
                    partial_derivative_dirname = MODEL_PARTIAL_DERIVATIVE_DIRNAME.format(partial_derivative, h_factor)
                    partial_derivative_dir = os.path.join(derivative_dir, partial_derivative_dirname)
                    
                    update_job_options_in_run_dirs(partial_derivative_dir, update_function)


def update_job_options_in_run_dirs(run_dir_path, update_function):  
    from ndop.model.constants import MODEL_RUN_DIRNAME
    
    runs_len = len(util.io.fs.get_dirs(run_dir_path))
    
    for run in range(runs_len):
        run_dirname = MODEL_RUN_DIRNAME.format(run)
        run_dir = os.path.join(run_dir_path, run_dirname)
        
        if os.path.exists(run_dir):
            update_job_options_in_job_options_dir(run_dir, update_function)


def update_job_options_in_job_options_dir(job_options_dir, update_function):
    logger.info('Updating job options in {}.'.format(job_options_dir))
    
    options_file = os.path.join(job_options_dir, 'job_options.hdf5')
    
    os.chmod(options_file, stat.S_IRUSR | stat.S_IWUSR)
    update_function(job_options_dir)
    os.chmod(options_file, stat.S_IRUSR)


## specific update functions

def update_output_dir():
    def update_function(job_options_dir):
        with ndop.model.job.Metos3D_Job(job_options_dir, force_load=True) as job:
            job.update_output_dir(job_options_dir)
    
    update_job_options(update_function)


def add_finished_file():
    def update_function(job_options_dir):
        options_file = os.path.join(job_options_dir, 'job_options.hdf5')
        
        with util.options.Options(options_file) as options:
            try:
                options['/job/finished_file']
                print('Finished file option already there in job option file {}.'.format(options_file))
            except KeyError:
                finished_file = os.path.join(job_options_dir, 'finished.txt')
                options['/job/finished_file'] = finished_file
                print('Finished file option added to job option file {}.'.format(options_file))
    
    update_job_options(update_function)


if __name__ == "__main__":
    with util.logging.Logger():
        add_finished_file()
    print('Update completed.')   