import os
import stat
import logging

from ndop.model.job import Metos3D_Job

import util.io
# import util.pattern
import util.logging



def update_job_options(time_step_sizes=(1,4,12,48)):
    from ndop.model.constants import MODEL_OUTPUT_DIR, MODEL_TIME_STEP_DIRNAME, MODEL_PARAMETERS_SET_DIRNAME, MODEL_SPINUP_DIRNAME, MODEL_RUN_DIRNAME, MODEL_DERIVATIVE_DIRNAME, MODEL_PARTIAL_DERIVATIVE_DIRNAME
    
    partial_derivatives = range(7)
    h_factors = (-1, 1)
    
    for time_step_size in time_step_sizes:
#         time_step_dirname = util.pattern.replace_int_pattern(MODEL_TIME_STEP_DIRNAME, time_step_size)
        time_step_dirname = MODEL_TIME_STEP_DIRNAME.format(time_step_size)
        time_step_dir = os.path.join(MODEL_OUTPUT_DIR, time_step_dirname)
        
        parameter_sets_len = len(util.io.get_dirs(time_step_dir))
        logging.info('{} parameter set dirs found in {}.'.format(parameter_sets_len, time_step_dir))
        
        for parameter_set_number in range(parameter_sets_len):
#             parameter_set_dirname = util.pattern.replace_int_pattern(MODEL_PARAMETERS_SET_DIRNAME, parameter_set_number)
            parameter_set_dirname = MODEL_PARAMETERS_SET_DIRNAME.format(parameter_set_number)
            parameter_set_dir = os.path.join(time_step_dir, parameter_set_dirname)
            
            spinup_dir = os.path.join(parameter_set_dir, MODEL_SPINUP_DIRNAME)
            
            update_from_run_dir_path(spinup_dir)
            
            derivative_dir = os.path.join(parameter_set_dir, MODEL_DERIVATIVE_DIRNAME)
            
            for partial_derivative in partial_derivatives:
                for h_factor in h_factors:
#                     partial_derivative_dirname = util.pattern.replace_int_pattern(MODEL_PARTIAL_DERIVATIVE_DIRNAME, (partial_derivative, h_factor_index))
                    partial_derivative_dirname = MODEL_PARTIAL_DERIVATIVE_DIRNAME.format(partial_derivative, h_factor)
                    partial_derivative_dir = os.path.join(derivative_dir, partial_derivative_dirname)
                    
                    update_from_run_dir_path(partial_derivative_dir)



def update_from_run_dir_path(run_dir_path):  
    from ndop.model.constants import MODEL_RUN_DIRNAME
    
    logging.debug('Checking runs in {}.'.format(run_dir_path))
    
    runs_len = len(util.io.get_dirs(run_dir_path))
    
    for run in range(runs_len):
#         run_dirname = util.pattern.replace_int_pattern(MODEL_RUN_DIRNAME, run)
        run_dirname = MODEL_RUN_DIRNAME.format(run)
        run_dir = os.path.join(run_dir_path, run_dirname)
        
        if os.path.exists(run_dir):
            update_from_job_option_path(run_dir)


def update_from_job_option_path(job_option_path):
    job_file = os.path.join(job_option_path, 'job_options.hdf5')
    
    logging.info('Updating job option file {}.'.format(job_file))
    
    os.chmod(job_file, stat.S_IRUSR | stat.S_IWUSR)
    
    with Metos3D_Job(output_path=job_option_path, force_load=True) as job:
#         job.load(job_file)
        job.update_output_path(job_option_path)
    
    os.chmod(job_file, stat.S_IRUSR)
    
 
if __name__ == "__main__":
    with util.logging.Logger():
        update_job_options()
    print('Update completed.')   