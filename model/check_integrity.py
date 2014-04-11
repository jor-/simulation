import os
import warnings

import logging
logger = logging.getLogger(__name__)

from ndop.model.job import Metos3D_Job

# import util.pattern
import util.io

#TODO check read only for finished jobs
#TODO grep ERROR on job output

def check_job_file_integrity_spinup(spinup_dir):
    run_dirs = util.io.get_dirs(spinup_dir)
        
    for run_dir in run_dirs:
        ## check job file and if job started
        try:
            with Metos3D_Job(run_dir, force_load=True) as job:
                if not job.is_started():
                    warnings.warn('Job in ' + run_dir + ' is not started.')
            logger.debug('Job file in {} is okay.'.format(run_dir))
        except (OSError, IOError):
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
                
            
    


def check_job_file_integrity(time_step_size=1):
    from ndop.model.constants import MODEL_OUTPUT_DIR, MODEL_TIME_STEP_DIRNAME, MODEL_SPINUP_DIRNAME, MODEL_DERIVATIVE_DIRNAME, JOB_OPTIONS_FILENAME
    
#     time_step_dirname = util.pattern.replace_int_pattern(MODEL_TIME_STEP_DIRNAME, time_step_size)
    time_step_dirname = MODEL_TIME_STEP_DIRNAME.format(time_step_size)
    time_step_dir = os.path.join(MODEL_OUTPUT_DIR, time_step_dirname)
    
    parameter_set_dirs = util.io.get_dirs(time_step_dir)
    
    for parameter_set_dir in parameter_set_dirs:
        
        spinup_dir = os.path.join(parameter_set_dir, MODEL_SPINUP_DIRNAME)
        check_job_file_integrity_spinup(spinup_dir)
        
        derivative_dir = os.path.join(parameter_set_dir, MODEL_DERIVATIVE_DIRNAME)
        partial_derivative_dirs = util.io.get_dirs(derivative_dir)
        for partial_derivative_dir in partial_derivative_dirs:
            check_job_file_integrity_spinup(partial_derivative_dir)
        


if __name__ == "__main__":
    check_job_file_integrity()
    print('Check completed.')