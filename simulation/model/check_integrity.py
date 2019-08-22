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
            base_dirs.append(model_dir)
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
                job.check_integrity(force_to_be_started=True, force_to_be_readonly=True)
        except util.batch.universal.system.JobError as e:
            util.logging.error(e)


def check_job_options(model_names=None):
    base_dirs = get_base_dirs(model_names=model_names)
    for base_dir in base_dirs:
        check_job_options_in_dir(base_dir)


def check_file_stats(model_names=None, owner=True, user=True, group=False, other=False):
    util.logging.info('Checking file stats with owner={owner}, user={user}, group={group}, other={other}.'.format(
        owner=owner, user=user, group=group, other=other))

    if owner:
        owner_uid = os.getuid()

    # check files
    def check_file(file):
        file_stat = os.stat(file)

        if owner and file_stat[stat.ST_UID] != owner_uid:
            util.logging.error('File {} has wrong owner!'.format(file))

        permissions = file_stat[stat.ST_MODE]
        if user is not None:
            if bool(permissions & stat.S_IRUSR) != user:
                util.logging.error('File {} should be readable {} for user, but it is not!'.format(file, user))
        if group is not None:
            if bool(permissions & stat.S_IRGRP) != group:
                util.logging.error('File {} should be readable {} for group, but it is not!'.format(file, group))
        if other is not None:
            if bool(permissions & stat.S_IROTH) != other:
                util.logging.error('File {} should be readable {} for other, but it is not!'.format(file, other))

    # check directory
    def check_dir(file):
        file_stat = os.stat(file)

        if owner and file_stat[stat.ST_UID] != owner_uid:
            util.logging.error('Directory {} has wrong owner!'.format(file))

        permissions = file_stat[stat.ST_MODE]
        if user is not None:
            if bool(permissions & stat.S_IRUSR) and bool(permissions & stat.S_IXUSR) != user:
                util.logging.error('Directory {} should be readable {} for user, but it is not!'.format(file, user))
        if group is not None:
            if bool(permissions & stat.S_IRGRP) and bool(permissions & stat.S_IXGRP) != group:
                util.logging.error('Directory {} should be readable {} for group, but it is not!'.format(file, group))
        if group is not None:
            if bool(permissions & stat.S_IROTH) and bool(permissions & stat.S_IXOTH) != other:
                util.logging.error('Directory {} should be readable {} for other, but it is not!'.format(file, other))

    for base_dir in get_base_dirs(model_names=model_names):
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
    parser.add_argument('--check_job_options', action='store_true')
    parser.add_argument('--check_database', action='store_true')
    parser.add_argument('--check_owner', action='store_true')
    parser.add_argument('--check_permissions', type=int, default=None, nargs='*')
    parser.add_argument('--model_names', type=str, default=None, choices=simulation.model.constants.MODEL_NAMES, nargs='+', help='The models to check. If not specified all models are checked')
    parser.add_argument('--check_job_options_in_dirs', '--dirs', type=str, default=None, nargs='+', help='The directories where to check job options files. If not specified no special directories are checked')
    parser.add_argument('--debug_level', choices=util.logging.LEVELS, default='INFO', help='Print debug infos low to passed level.')
    parser.add_argument('--version', action='version', version='%(prog)s {}'.format(simulation.__version__))
    args = parser.parse_args()

    # call function
    with util.logging.Logger(level=args.debug_level):
        if args.check_job_options:
            check_job_options(model_names=args.model_names)
        if args.check_owner or args.check_permissions:
            user = None
            group = None
            other = None
            check_permissions = args.check_permissions
            if check_permissions is not None:
                check_permissions = tuple(bool(p) for p in check_permissions)
                if len(check_permissions) == 0:
                    user = True
                else:
                    user = check_permissions[0]
                if len(check_permissions) >= 2:
                    group = check_permissions[1]
                if len(check_permissions) >= 3:
                    other = check_permissions[2]
            check_file_stats(model_names=args.model_names, owner=args.check_owner, user=user, group=group, other=other)
        if args.check_database:
            check_db(model_names=args.model_names)
        if args.check_job_options_in_dirs is not None:
            for job_options_in_dir in args.check_job_options_in_dirs:
                check_job_options_in_dir(job_options_in_dir)
        util.logging.info('Check completed.')


if __name__ == "__main__":
    _main()
