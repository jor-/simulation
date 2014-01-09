import os
import stat

from ndop.metos3d.job import Metos3D_Job

import util.pattern
from util.debug import print_debug



def update_job_options(time_step_sizes=(1,4,12,48), parameter_sets=range(1000), debug_level=0, required_debug_level=1):
    from ndop.metos3d.constants import MODEL_OUTPUTS_DIR, MODEL_TIME_STEP_DIRNAME, MODEL_PARAMETERS_SET_DIRNAME, MODEL_SPINUP_DIRNAME, MODEL_RUN_DIRNAME, MODEL_DERIVATIVE_DIRNAME, MODEL_PARTIAL_DERIVATIVE_DIRNAME
    
    partial_derivatives = range(7)
    h_factors = range(2)
    
    for time_step_size in time_step_sizes:
        time_step_dirname = util.pattern.replace_int_pattern(MODEL_TIME_STEP_DIRNAME, time_step_size)
        time_step_dir = os.path.join(MODEL_OUTPUTS_DIR, time_step_dirname)
        
        for parameter_set_number in parameter_sets:
            parameter_set_dirname = util.pattern.replace_int_pattern(MODEL_PARAMETERS_SET_DIRNAME, parameter_set_number)
            parameter_set_dir = os.path.join(time_step_dir, parameter_set_dirname)
            
            spinup_dir = os.path.join(parameter_set_dir, MODEL_SPINUP_DIRNAME)
            
            update_from_run_dir_path(spinup_dir, debug_level=debug_level, required_debug_level=required_debug_level)
            
            derivative_dir = os.path.join(parameter_set_dir, MODEL_DERIVATIVE_DIRNAME)
            
            for partial_derivative in partial_derivatives:
                for h_factor in h_factors:
                    partial_derivative_dirname = util.pattern.replace_int_pattern(MODEL_PARTIAL_DERIVATIVE_DIRNAME, (partial_derivative, h_factor))
                    partial_derivative_dir = os.path.join(derivative_dir, partial_derivative_dirname)
                    
                    update_from_run_dir_path(spinup_dir, debug_level=debug_level, required_debug_level=required_debug_level)



def update_from_run_dir_path(run_dir_path, debug_level=0, required_debug_level=1):  
    from ndop.metos3d.constants import MODEL_RUN_DIRNAME
    
    runs = range(10)
    
    for run in runs:
        run_dirname = util.pattern.replace_int_pattern(MODEL_RUN_DIRNAME, run)
        run_dir = os.path.join(run_dir_path, run_dirname)
        
        if os.path.exists(run_dir):
            update_from_job_option_path(run_dir, debug_level=debug_level, required_debug_level=required_debug_level)


def update_from_job_option_path(job_option_path, debug_level=0, required_debug_level=1):
    job_file = os.path.join(job_option_path, 'job_options.hdf5')
    
    print_debug(('Updating job option file "', job_file, '".'), debug_level, required_debug_level)
    
    os.chmod(job_file, stat.S_IRUSR | stat.S_IWUSR)
    
    with Metos3D_Job() as job:
        job.load(job_file)
        job.update_output_path(job_option_path)
    
    os.chmod(job_file, stat.S_IRUSR)
    
    