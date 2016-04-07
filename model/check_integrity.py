import argparse
import os
import stat
import numpy as np

import simulation.model.eval
import simulation.model.job
import simulation.model.constants
import simulation.util.data_base
import simulation.constants

import util.options
import util.io.fs
import util.batch.universal.system
import util.index_database.general

#TODO check read only for finished jobs
#TODO check cache option files available
#TODO at multiple runs check if right successor

ERROR_IGNORE_LIST = ("librdmacm: Fatal: unable to get RDMA device list"+os.linesep, "librdmacm: Warning: couldn't read ABI version."+os.linesep, "librdmacm: Warning: assuming: 4"+os.linesep, 'cpuinfo: error while loading shared libraries: libgcc_s.so.1: cannot open shared object file: No such file or directory'+os.linesep)

def check_db_entry_integrity_spinup(spinup_dir, is_spinup):
    
    
    run_dirs = util.io.fs.get_dirs(spinup_dir)
    run_dirs.sort()
    n = len(run_dirs)

    if n == 0:
        print('No run dirs in ' + spinup_dir + '.')
    else:
        
        ## check run dirs
        for run_dir_index in range(n):
            run_dir = run_dirs[run_dir_index]

            ## check if dirs in run dir exist
            dirs = util.io.fs.get_dirs(run_dir)
            if len(dirs) > 0:
                print('Directories found in {}.'.format(run_dir))


            ## check job file
            try:
                with simulation.model.job.Metos3D_Job(run_dir, force_load=True) as job:
                    ## check if started
                    if not job.is_started():
                        print('Job in {} is not started!'.format(run_dir))
                    try:
                        is_running = job.is_running()
                    except util.batch.universal.system.JobError as e:
                        print(e)
                        break
                    job_output_file = job.output_file
                    try:
                        job_id = job.id
                    except Exception:
                        print('Job in {} is not started!'.format(run_dir))
                        break
                    
                    ## check read only
                    if not job.options.is_read_only():
                        print('Job option file in {} is writeable!'.format(run_dir))
                    
                    ## check options
                    options_file = os.path.join(run_dir, 'job_options.hdf5')
                    options = util.options.Options(options_file, replace_environment_vars_at_set=False, replace_environment_vars_at_get=False)
                    file_entry_prefix = '${{{}}}'.format(simulation.constants.SIMULATION_OUTPUT_DIR_ENV_NAME)
                    must_have_tracer_input = not is_spinup or run_dir_index > 0
                    for file_key, must_exists in [('/job/id_file', True), ('/job/option_file', True), ('/job/output_file', True), ('/job/finished_file', True), ('/job/unfinished_file', True), ('/model/tracer_input_dir', must_have_tracer_input), ('/metos3d/tracer_input_dir', must_have_tracer_input), ('/metos3d/output_dir', True), ('/metos3d/option_file', True)]:
                        try:
                            value = options[file_key]
                        except KeyError:
                            if must_exists:
                                print('Job option {} in {} is missing.'.format(file_key, run_dir))
                        else:
                            if not value.startswith(file_entry_prefix):
                                print('Job option {} in {} is not okay. It should start with {} but its is {}.'.format(file_key, run_dir, file_entry_prefix, value))
                    
                    ## check tracer input
                    if must_have_tracer_input:
                        if not simulation.model.constants.DATABASE_SPINUP_DIRNAME in options['/model/tracer_input_dir']:
                            print('Model tracer input dir {} in job file in {} is not a spinup run dir.'.format(options['/model/tracer_input_dir'], run_dir))
                        if options['/metos3d/tracer_input_dir'] != options['/metos3d/output_dir']:
                            print('Metos3D tracer input dir {} and tracer output dir in job file in {} are not the same.'.format(options['/metos3d/tracer_input_dir'], options['/metos3d/output_dir'], run_dir))
                            
                            
            except (OSError, IOError):
                print('Job file in {} is not okay.'.format(run_dir))
                break


            ## check petsc input files
            for input_filename in ('dop_input.petsc', 'po4_input.petsc') :
                input_file = os.path.join(run_dir, input_filename)
                if run_dir_index == 0 and is_spinup:
                    if os.path.exists(input_file) or os.path.lexists(input_file):
                        print('Petsc input files for run index == 0 found in {}!'.format(run_dir))
                        break
                else:
                    if not os.path.lexists(input_file):
                        if is_spinup:
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
                with simulation.model.job.Metos3D_Job(run_dir, force_load=True) as job:
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
                    except:
                        print('The job output file {} could not be opened!'.format(job_output_file))
                else:
                    print('Job output file {} does not exist!'.format(job_output_file))

                with simulation.model.job.Metos3D_Job(run_dir, force_load=True) as job:
                    try:
                        job.last_year
                    except:
                        print('The job output file {} format is not correct! Last year could not be computed'.format(job_output_file))
                    try:
                        job.last_tolerance
                    except:
                        print('The job output file {} format is not correct! Last tolerance could not be computed'.format(job_output_file))
        



def check_db_entry_integrity(model_name='dop_po4', time_step=1, parameter_set_dirs_to_check=None, check_for_same_parameters=True):
    from simulation.model.constants import DATABASE_OUTPUT_DIR, DATABASE_MODEL_DIRNAME, DATABASE_TIME_STEP_DIRNAME, DATABASE_SPINUP_DIRNAME, DATABASE_DERIVATIVE_DIRNAME, JOB_OPTIONS_FILENAME, DATABASE_PARAMETERS_FILENAME
    from simulation.util.constants import CACHE_DIRNAME, WOD_F_FILENAME, WOD_DF_FILENAME

    wod_m = simulation.util.data_base.WOD().m

    model_dirname = DATABASE_MODEL_DIRNAME.format(model_name)
    model_dir = os.path.join(DATABASE_OUTPUT_DIR, model_dirname)
    time_step_dirname = DATABASE_TIME_STEP_DIRNAME.format(time_step)
    time_step_dir = os.path.join(model_dir, time_step_dirname)
    df_step_sizes = [10**(-6), 10**(-7)]

    check_all_parameter_sets = parameter_set_dirs_to_check is None or (len(parameter_set_dirs_to_check) == 1 and parameter_set_dirs_to_check[0] is None)
    check_for_same_parameters = check_for_same_parameters and (check_all_parameter_sets or len(parameter_set_dirs_to_check) > 1)
    if check_all_parameter_sets or check_for_same_parameters:
        parameter_set_dirs_all = util.io.fs.get_dirs(time_step_dir)
    if check_all_parameter_sets:
        parameter_set_dirs_to_check = parameter_set_dirs_all
    # if parameter_set_dirs_to_check is None or (len(parameter_set_dirs_to_check) == 1 and parameter_set_dirs_to_check[0] is None):
    #     parameter_set_dirs_to_check = parameter_set_dirs_all

    for parameter_set_dir in parameter_set_dirs_to_check:
    
        print('Checking integrity of parameter set {}.'.format(parameter_set_dir))
        
        ## check spinup dir
        spinup_dir = os.path.join(parameter_set_dir, DATABASE_SPINUP_DIRNAME)
        check_db_entry_integrity_spinup(spinup_dir, True)
        
        ## check derivative dir
        for df_step_size in df_step_sizes:
            derivative_dir = os.path.join(parameter_set_dir, DATABASE_DERIVATIVE_DIRNAME.format(df_step_size))
            partial_derivative_dirs = util.io.fs.get_dirs(derivative_dir)
            for partial_derivative_dir in partial_derivative_dirs:
                check_db_entry_integrity_spinup(partial_derivative_dir, False)

        ## check for parameters
        p = np.loadtxt(os.path.join(parameter_set_dir, DATABASE_PARAMETERS_FILENAME))
        if not np.all(np.isfinite(p)):
            print('Parameters {} in set {} are not finite!'.format(p, parameter_set_dir))

        ## check for same parameters
        if check_for_same_parameters:
            for parameter_set_dir_i in parameter_set_dirs_all:
                if parameter_set_dir_i != parameter_set_dir:
                    p_i = np.loadtxt(os.path.join(parameter_set_dir_i, DATABASE_PARAMETERS_FILENAME))
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

        ## check value cache
        value_cache_option_files = util.io.fs.filter_files(parameter_set_dir, lambda s: s.endswith('options.npy'), recursive=True)
        for value_cache_option_file in value_cache_option_files:
            value_cache_option = np.load(value_cache_option_file)
            if not value_cache_option.ndim == 1:
                print('Value cache option {} has ndim {}!'.format(value_cache_option_file, value_cache_option.ndim))
            if not len(value_cache_option) in [3, 6]:
                print('Value cache option {} has len {}!'.format(value_cache_option_file, len(value_cache_option)))
        
        ## check file permissions
        def check_file(file):
            permissions = os.stat(file)[stat.ST_MODE]
            if not (permissions & stat.S_IRUSR and permissions & stat.S_IRGRP):
                print('File {} is not readable!'.format(file))
        def check_dir(file):
            permissions = os.stat(file)[stat.ST_MODE]
            if not (permissions & stat.S_IRUSR and permissions & stat.S_IXUSR and permissions & stat.S_IRGRP and permissions & stat.S_IXGRP):
                print('Dir {} is not readable!'.format(file))
        
        util.io.fs.walk_all_in_dir(parameter_set_dir, check_file, check_dir, exclude_dir=False, topdown=True)



def check_db_integrity(model_name='dop_po4', time_step=1):
    print('Checking parameter database integrity.')
    
    model_options = {'time_step': time_step}
    m = simulation.model.eval.Model(model_options=model_options)
    array_db = m._parameter_db.array_db
    file_db = m._parameter_db.file_db
    
    for index in array_db.used_indices():
        v_a = array_db.get_value(index)
        try:
            v_f = file_db.get_value(index)
        except util.index_database.general.DatabaseIndexError:
            print('Array db hast value at index {} and file db has not value their!'.format(index))
        else:
            if not array_db.are_values_equal(v_a, v_f):
                print('Array db and file db value at index {} are not equal: {} != {}!'.format(index, v_a, v_f))



if __name__ == "__main__":
    ## configure arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--skip_same_parameter_check', action='store_true')
    parser.add_argument('parameter_set_dir', nargs='?', default=None)
    args = parser.parse_args()
    ## run check
    time_step = 1
    if args.parameter_set_dir is None:
        check_db_integrity(time_step=time_step)
    check_db_entry_integrity(time_step=time_step, parameter_set_dirs_to_check=(args.parameter_set_dir,), check_for_same_parameters=not args.skip_same_parameter_check)
    print('Check completed.')