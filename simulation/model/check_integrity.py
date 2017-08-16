import os
import stat

import simulation
import simulation.model.job
import simulation.model.constants
import simulation.model.cache
import simulation.model.options

import util.io.fs
import util.batch.universal.system
import util.index_database.general
import util.logging


# util functions

def get_base_dirs(model_names=None):
    if model_names is None:
        database_dir = simulation.model.constants.DATABASE_OUTPUT_DIR
        base_dirs = [database_dir]
    else:
        base_dirs = []
        for model_name in model_names:
            model_dirname = simulation.model.constants.DATABASE_MODEL_DIRNAME.format(model_name)
            model_dir = os.path.join(simulation.model.constants.DATABASE_OUTPUT_DIR, model_dirname)
            base_dirs.extend(model_dir)
    return base_dirs


def get_files_in_dir(pattern, directory):
    util.logging.info('Getting jobs in {}.'.format(directory))
    files = util.io.fs.get_files(directory, filename_pattern=pattern, use_absolute_filenames=True, recursive=True)
    util.logging.info('Got {} jobs.'.format(len(files)))
    return files


def get_files(pattern, model_names=None):
    files = []
    base_dirs = get_base_dirs(model_names=model_names)
    for base_dir in base_dirs:
        files.extend(get_files_in_dir(pattern, base_dir))
    return files


# check functions

def check_job_options_in_dir(directory):
    util.logging.info('Checking job options integrity in {}.'.format(directory))

    pattern = '*/job_options.hdf5'
    files = get_files_in_dir(pattern, directory)
    for file in files:
        run_dir = os.path.dirname(file)
        try:
            with simulation.model.job.Metos3D_Job(run_dir, force_load=True) as job:
                job.check_integrity(should_be_started=True, should_be_readonly=True)
        except Exception as e:
            util.logging.error(e)


def check_job_options(model_names=None):
    base_dirs = get_base_dirs(model_names=model_names)
    for base_dir in base_dirs:
        check_job_options_in_dir(base_dir)


def check_permissions(model_names=None):
    util.logging.info('Checking permissions.')

    base_dirs = get_base_dirs(model_names=model_names)

    def check_file(file):
        permissions = os.stat(file)[stat.ST_MODE]
        if not (permissions & stat.S_IRUSR and permissions & stat.S_IRGRP):
            util.logging.error('File {} is not readable!'.format(file))

    def check_dir(file):
        permissions = os.stat(file)[stat.ST_MODE]
        if not (permissions & stat.S_IRUSR and permissions & stat.S_IXUSR and permissions & stat.S_IRGRP and permissions & stat.S_IXGRP):
            util.logging.error('Dir {} is not readable!'.format(file))

    for base_dir in base_dirs:
        util.io.fs.walk_all_in_dir(base_dir, check_file, check_dir, exclude_dir=False, topdown=True)


def check_db(model_names=None):
    util.logging.info('Checking database integrity.')

    model = simulation.model.cache.Model()
    try:
        model.check_integrity(model_names=model_names)
    except (util.index_database.general.DatabaseError, simulation.model.eval.DatabaseError, OSError) as e:
        util.logging.error(e)


# *** main function for script call *** #

def _main():

    # parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--check_job_options', action='store_true')
    parser.add_argument('-p', '--check_permissions', action='store_true')
    parser.add_argument('-d', '--check_database', action='store_true')
    parser.add_argument('-m', '--model_names', default=None, nargs='+', help='The models to check. If not specified all models are checked')
    parser.add_argument('-f', '--check_job_options_in_dir', default=None, nargs='+', help='The directories where to check job options files. If not specified no special directories are checked')
    parser.add_argument('-D', '--debug_level', choices=util.logging.LEVELS, default='INFO', help='Print debug infos low to passed level.')
    parser.add_argument('--version', action='version', version='%(prog)s {}'.format(simulation.__version__))
    args = parser.parse_args()

    # call function
    with util.logging.Logger(level=args.debug_level):
        if args.check_job_options:
            check_job_options(model_names=args.model_names)
        if args.check_permissions:
            check_permissions(model_names=args.model_names)
        if args.check_database:
            check_db(model_names=args.model_names)
        util.logging.info('Check completed.')


if __name__ == "__main__":
    _main()
