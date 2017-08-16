import os

import numpy as np

import util.io.fs
import util.petsc.universal
import simulation.model.job

def convert(run_dir):
    with simulation.model.job.Metos3D_Job(run_dir, force_load=True) as job:
        try:
            model_tracer_input_files = job.model_tracer_input_files
        except KeyError:
            model_tracer_input_files = None
            print('{} has no input files.'.format(run_dir))
        else:
            metos_tracer_input_files = job.tracer_input_files
    
    if model_tracer_input_files is not None:
        n = len(model_tracer_input_files)
        for i in range(n):
            metos_tracer_input_file = metos_tracer_input_files[i]
            assert run_dir == os.path.dirname(metos_tracer_input_file)
            model_tracer_input_file = model_tracer_input_files[i]
            if not os.path.islink(metos_tracer_input_file):
                metos_array = util.petsc.universal.load_petsc_vec_to_numpy_array(metos_tracer_input_file)
                model_array = util.petsc.universal.load_petsc_vec_to_numpy_array(model_tracer_input_file)
                if np.all(metos_array == model_array):
                    model_tracer_input_relative_file = os.path.relpath(model_tracer_input_file, start=run_dir)
                    print('Symlinking {} in {}.'.format(model_tracer_input_relative_file, metos_tracer_input_file))
                    util.io.fs.remove_file(metos_tracer_input_file, force=True)
                    os.symlink(model_tracer_input_relative_file, metos_tracer_input_file)
                else:
                    print('Do not symlink {}, because it is not equal {}.'.format(metos_tracer_input_file, model_tracer_input_file))
            else:
                print('{} is already a symlink.'.format(metos_tracer_input_file))


def update():
    database_dir = simulation.model.constants.DATABASE_OUTPUT_DIR
    database_dir = os.path.join(database_dir, 'model_MITgcm-PO4-DOP')
    run_dirs = util.io.fs.get_files(database_dir, filename_pattern='*/job_options.hdf5', use_absolute_filenames=True, recursive=True)
    run_dirs = [os.path.dirname(job_file) for job_file in run_dirs]
    print('Found {} run dirs.'.format(len(run_dirs)))
    for run_dir in run_dirs:
        convert(run_dir)


if __name__ == '__main__':
    update()
    print('Update completed.')