import os

from ndop.model.job import Metos3D_Job

import util.io
import util.rzcluster.interact

#TODO check read only for finished jobs

def check_job_file_integrity_spinup(spinup_dir, is_spinup_dir):
    run_dirs = util.io.get_dirs(spinup_dir)
    n = len(run_dirs)
    
    if n == 0:
        print('No run dirs in ' + spinup_dir + '.')
    else:
        for run_dir_index in range(n):
            run_dir = run_dirs[run_dir_index]
            
            ## check job file
            try:
                with Metos3D_Job(run_dir, force_load=True) as job:
                    if not job.is_started():
                        print('Job in ' + run_dir + ' is not started.')
                    is_running = job.is_running()
                    job_output_file = job.output_file
                    try:
                        job_id = job.id
                    except Exception:
                        print('Job in {} is not started!'.format(run_dir))
                        break
            except (OSError, IOError):
                print('Job file in ' + run_dir + ' is not okay.')
                is_running = False
                job_output_file = None
            
            ## check if trajectory dir exist
            trajectory_dirs = util.io.get_dirs(run_dir)
            if len(trajectory_dirs) > 0:
                print('Trajectory directories found: ' + ' '.join(trajectory_dirs))
            
            ## check petsc input files
            for input_filename in ('dop_input.petsc', 'po4_input.petsc') :
                input_file = os.path.join(run_dir, input_filename)
                if run_dir_index == 0 and is_spinup_dir:
                    if os.path.exists(input_file) or os.path.lexists(input_file):
                        print('Petsc input files for run index == 0 found in {}!'.format(run_dir))
                        break
                else:
                    if not os.path.lexists(input_file):
                        if is_spinup_dir:
                            print('No petsc input files for run index > 0 found in {}!'.format(run_dir))
                        else:
                            print('No petsc input files for derivative run found in {}!'.format(run_dir))
                        break
                    elif not os.path.exists(input_file):
                        print('Link for petsc input files for run index > 0 found in {} broken!'.format(run_dir))
                        break
            
            
            ## check if petsc output files exist
            output_files_exist = True
            for petsc_output_filename in ('dop_output.petsc', 'po4_output.petsc') :
                petsc_output_file = os.path.join(run_dir, petsc_output_filename)
                output_files_exist = output_files_exist and os.path.exists(petsc_output_file)
            
            ## check finish files
            finished_file = os.path.join(run_dir, 'finished.txt')
            if output_files_exist and not os.path.exists(finished_file):
                print('Output files exist but finished file {} does not exits!'.format(finished_file))
                break
            
            ## check petsc output files
            if is_running:
                if output_files_exist:
                    print('Job is running but output files in {} do exist!'.format(run_dir))
                    break
                else:
                    if util.rzcluster.interact.is_job_finished(job_id):
                        print('Job in run dir {} should run, but it is not!'.format(run_dir))
                    if run_dir_index != n-1:
                        print('Job in run dir {} not finished, but it has not the last run index!'.format(run_dir))
                        break
            else:
                if not output_files_exist:
                    print('Output files in {} do not exist!'.format(run_dir))
                    break
            
            
            ## check job output file exists
            if not is_running and not os.path.exists(job_output_file):
                print('Job output file {} does not exist!'.format(job_output_file))
                
            
            ## check job output file for errors
            if job_output_file is not None:
                try:
                    with open(job_output_file) as output_file_object:
                        for line in output_file_object:
                            if "ERROR" in line:
                                print('There are errors in the job output file {}: {}.'.format(job_output_file, line))
                                break
                except e:
                    print('The job output file {} could not be opened!'% job_output_file)
            
    


def check_job_file_integrity(time_step_size=1):
    from ndop.model.constants import MODEL_OUTPUT_DIR, MODEL_TIME_STEP_DIRNAME, MODEL_SPINUP_DIRNAME, MODEL_DERIVATIVE_DIRNAME, JOB_OPTIONS_FILENAME
    
#     time_step_dirname = util.pattern.replace_int_pattern(MODEL_TIME_STEP_DIRNAME, time_step_size)
    time_step_dirname = MODEL_TIME_STEP_DIRNAME.format(time_step_size)
    time_step_dir = os.path.join(MODEL_OUTPUT_DIR, time_step_dirname)
    
    parameter_set_dirs = util.io.get_dirs(time_step_dir)
    
    for parameter_set_dir in parameter_set_dirs:
        
        spinup_dir = os.path.join(parameter_set_dir, MODEL_SPINUP_DIRNAME)
        check_job_file_integrity_spinup(spinup_dir, True)
        
        derivative_dir = os.path.join(parameter_set_dir, MODEL_DERIVATIVE_DIRNAME)
        partial_derivative_dirs = util.io.get_dirs(derivative_dir)
        for partial_derivative_dir in partial_derivative_dirs:
            check_job_file_integrity_spinup(partial_derivative_dir, False)
        


if __name__ == "__main__":
    check_job_file_integrity()
    print('Check completed.')