import argparse
import os
import numpy as np

from ndop.model.job import Metos3D_Job
import ndop.util.data_base

import util.io.fs
import util.batch.universal.system

#TODO check read only for finished jobs
#TODO check cache option files available

ERROR_IGNORE_LIST = ("librdmacm: Fatal: unable to get RDMA device list"+os.linesep, "librdmacm: Warning: couldn't read ABI version."+os.linesep, "librdmacm: Warning: assuming: 4"+os.linesep)

def check_job_file_integrity_spinup(spinup_dir, is_spinup_dir):
    run_dirs = util.io.fs.get_dirs(spinup_dir)
    n = len(run_dirs)

    if n == 0:
        print('No run dirs in ' + spinup_dir + '.')
    else:
        for run_dir_index in range(n):
            run_dir = run_dirs[run_dir_index]


            ## check if dirs in run dir exist
            dirs = util.io.fs.get_dirs(run_dir)
            if len(dirs) > 0:
                print('Directories found in {}.'.format(run_dir))


            ## check job file
            try:
                with Metos3D_Job(run_dir, force_load=True) as job:
                    ## check if started
                    if not job.is_started():
                        print('Job in {} is not started!'.format(run_dir))
                    is_running = job.is_running()
                    job_output_file = job.output_file
                    try:
                        job_id = job.id
                    except Exception:
                        print('Job in {} is not started!'.format(run_dir))
                        break
                    ## check read only
                    if not job.options.is_read_only():
                        print('Job option file in {} is writeable!'.format(run_dir))
            except (OSError, IOError):
                print('Job file in ' + run_dir + ' is not okay.')
                break


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
            petsc_output_files_exist = []
            for petsc_output_filename in ('dop_output.petsc', 'po4_output.petsc'):
                petsc_output_file = os.path.join(run_dir, petsc_output_filename)
                petsc_output_files_exist.append(os.path.exists(petsc_output_file))

            if not np.all(petsc_output_files_exist) and (run_dir_index != n-1 or not is_running):
                if run_dir_index != n-1:
                    print('Petsc output files in {} do not exist, but it has not the last run index!'.format(run_dir))
                else:
                    print('Petsc output files in {} do not exist, but the job is not started or finished!'.format(run_dir))
                break


            ## check finish file
            finished_file = os.path.join(run_dir, 'finished.txt')
            if np.any(petsc_output_files_exist) and not os.path.exists(finished_file):
                print('Petsc output files in {} exist but finished file does not exist!'.format(run_dir))
                break


            if is_running:
                ## check if really running
                try:
                    is_really_running = util.batch.universal.system.BATCH_SYSTEM.is_job_running(job_id)
                    if not is_really_running:
                        print('Job in {} should run but it does not!'.format(run_dir))
                        break
                except ConnectionError:
                    print('Cannot connect to job server. Please check job id {}'.format(job_id))


                ## check if petsc output files exist
                if np.any(petsc_output_files_exist):
                    print('Job is running but petsc output files in {} do exist!'.format(run_dir))
                    break
            else:
                ## check exit code
                with Metos3D_Job(run_dir, force_load=True) as job:
                    exit_code = job.exit_code

                if exit_code != 0:
                    print('Job in {} has exit code {}!'.format(run_dir, exit_code))


                ## check job output file
                if os.path.exists(job_output_file):
                    try:
                        with open(job_output_file) as output_file_object:
                            for line in output_file_object:
                                line_lower = line.lower()
                                if ('error' in line_lower and not 'error_path' in line_lower) or 'warning' in line_lower or 'fatal' in line_lower or 'permission denied' in line_lower:
                                    if line not in ERROR_IGNORE_LIST:
                                        print('There are errors in the job output file {}: {}.'.format(job_output_file, line))
                                        break
                    except e:
                        print('The job output file {} could not be opened!'.format(job_output_file))
                else:
                    print('Job output file {} does not exist!'.format(job_output_file))




def check_job_file_integrity(time_step_size=1, parameter_set_dirs_to_check=None, check_for_same_parameters=True):
    from ndop.model.constants import MODEL_OUTPUT_DIR, MODEL_TIME_STEP_DIRNAME, MODEL_SPINUP_DIRNAME, MODEL_DERIVATIVE_DIRNAME, JOB_OPTIONS_FILENAME, MODEL_PARAMETERS_FILENAME
    from ndop.util.constants import CACHE_DIRNAME, WOD_F_FILENAME, WOD_DF_FILENAME

    wod_m = ndop.util.data_base.WOD().m

    time_step_dirname = MODEL_TIME_STEP_DIRNAME.format(time_step_size)
    time_step_dir = os.path.join(MODEL_OUTPUT_DIR, time_step_dirname)
    df_step_size = 10**(-7)

    parameter_set_dirs_all = util.io.fs.get_dirs(time_step_dir)
    if parameter_set_dirs_to_check is None or (len(parameter_set_dirs_to_check) == 1 and parameter_set_dirs_to_check[0] is None):
        parameter_set_dirs_to_check = parameter_set_dirs_all

    for parameter_set_dir in parameter_set_dirs_to_check:

        spinup_dir = os.path.join(parameter_set_dir, MODEL_SPINUP_DIRNAME)
        check_job_file_integrity_spinup(spinup_dir, True)

        derivative_dir = os.path.join(parameter_set_dir, MODEL_DERIVATIVE_DIRNAME.format(df_step_size))
        partial_derivative_dirs = util.io.fs.get_dirs(derivative_dir)
        for partial_derivative_dir in partial_derivative_dirs:
            check_job_file_integrity_spinup(partial_derivative_dir, False)


        ## check for parameters
        p = np.loadtxt(os.path.join(parameter_set_dir, MODEL_PARAMETERS_FILENAME))
        if not np.all(np.isfinite(p)):
            print('Parameters {} in set {} are not finite!'.format(p, parameter_set_dir))

        ## check for same parameters
        if check_for_same_parameters:
            for parameter_set_dir_i in parameter_set_dirs_all:
                if parameter_set_dir_i != parameter_set_dir:
                    p_i = np.loadtxt(os.path.join(parameter_set_dir_i, MODEL_PARAMETERS_FILENAME))
                    # if np.allclose(p, p_i):
                    if np.all(p == p_i):
                        print('Parameter set {} and {} have same parameters!'.format(parameter_set_dir, parameter_set_dir_i))

        ## check WOD output
        f_wod_file = os.path.join(parameter_set_dir, CACHE_DIRNAME, WOD_F_FILENAME)
        try:
            f_wod = np.load(f_wod_file)
        except FileNotFoundError:
            f_wod = None
        if f_wod is not None:
            if f_wod.ndim != 1 or len(f_wod) != wod_m:
                print('Wod f file {} has wrong shape {}!'.format(f_wod_file, f_wod.shape))

        df_wod_file = os.path.join(parameter_set_dir, CACHE_DIRNAME, WOD_DF_FILENAME)
        try:
            df_wod = np.load(df_wod_file)
        except FileNotFoundError:
            df_wod = None
        if df_wod is not None:
            if df_wod.ndim != 2 or len(df_wod) != wod_m or df_wod.shape[1] != len(p):
                print('Wod df file {} has wrong shape {}!'.format(df_wod_file, df_wod.shape))





if __name__ == "__main__":
    ## configure arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--skip_same_parameter_check', action='store_true')
    parser.add_argument('parameter_set_dir', nargs='?', default=None)
    args = parser.parse_args()
    ## run check
    check_job_file_integrity(parameter_set_dirs_to_check=(args.parameter_set_dir,), check_for_same_parameters=not args.skip_same_parameter_check)
    print('Check completed.')