import os
import warnings

from ndop.metos3d.job import Metos3D_Job

import util.pattern
import util.io
from util.debug import print_debug


def check_job_file_integrity_spinup(spinup_dir, debug_level=0, required_debug_level=1):
    run_dirs = util.io.get_dirs(spinup_dir)
        
    for run_dir in run_dirs:
        ## check job file and if job started
        try:
            with Metos3D_Job(run_dir, force_load=True) as job:
                if not job.is_started():
                    warnings.warn('Job in ' + run_dir + ' is not started.')
            print_debug(('Job file in ', run_dir, ' is okay.'), debug_level, required_debug_level)
        except:
            warnings.warn('Job file in ' + run_dir + ' is not okay.')
        
        ## check if trajectory dir exist
        trajectory_dirs = util.io.get_dirs(run_dir)
        if len(trajectory_dirs) > 0:
            warnings.warn('Trajectory directories found: ' + ','.join(trajectory_dirs))
        
        ## check if input files exist
        for input_filename in ('dop_input.petsc', 'po4_input.petsc') :
            input_file = os.path.join(run_dir, input_filename)
            if os.path.lexists(input_file) and not os.path.exists(input_file):
                warnings.warn('Link %s broken!'% input_file)
        
        ## check if output files exist
        for output_filename in ('dop_output.petsc', 'po4_output.petsc') :
            output_file = os.path.join(run_dir, output_filename)
            if not os.path.exists(output_file):
                warnings.warn('Output file %s does not exits!'% output_file)
                
            
    


def check_job_file_integrity(time_step_size=1, debug_level=0, required_debug_level=1):
    from ndop.metos3d.constants import MODEL_OUTPUTS_DIR, MODEL_TIME_STEP_DIRNAME, MODEL_SPINUP_DIRNAME, MODEL_DERIVATIVE_DIRNAME, JOB_OPTIONS_FILENAME
    
    time_step_dirname = util.pattern.replace_int_pattern(MODEL_TIME_STEP_DIRNAME, time_step_size)
    time_step_dir = os.path.join(MODEL_OUTPUTS_DIR, time_step_dirname)
    
    parameter_set_dirs = util.io.get_dirs(time_step_dir)
    
    for parameter_set_dir in parameter_set_dirs:
        
        spinup_dir = os.path.join(parameter_set_dir, MODEL_SPINUP_DIRNAME)
        check_job_file_integrity_spinup(spinup_dir, debug_level, required_debug_level+1)
        
        derivative_dir = os.path.join(parameter_set_dir, MODEL_DERIVATIVE_DIRNAME)
        partial_derivative_dirs = util.io.get_dirs(derivative_dir)
        for partial_derivative_dir in partial_derivative_dirs:
            check_job_file_integrity_spinup(partial_derivative_dir, debug_level, required_debug_level+1)
        


if __name__ == "__main__":
    check_job_file_integrity(debug_level=0)
    print('Check completed.')