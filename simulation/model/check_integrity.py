import argparse
import os
import stat
import numpy as np

import simulation.model.job
import simulation.model.constants
import simulation.model.cache
import simulation.model.options

import util.io.fs
import util.batch.universal.system
import util.index_database.general
import util.logging
logger = util.logging.logger


## util functions

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
    logger.info('Getting jobs in {}.'.format(directory))
    files = util.io.fs.get_files(directory, filename_pattern=pattern, use_absolute_filenames=True, recursive=True)
    logger.info('Got {} jobs.'.format(len(files)))
    return files
    

def get_files(pattern, model_names=None):
    files = []
    base_dirs = get_base_dirs(model_names=model_names)
    for base_dir in base_dirs:
        files.extend(get_files_in_dir(pattern, base_dir))
    return files


## check functions

def check_job_options(model_names=None):
    logger.info('Checking job options integrity.')
    
    pattern = '*/job_options.hdf5'
    files = get_files(pattern, model_names=model_names)
    for file in files:
        run_dir = os.path.dirname(file)
        try:
            with simulation.model.job.Metos3D_Job(run_dir, force_load=True) as job:
                job.check_integrity(should_be_started=True, should_be_readonly=True)
        except util.batch.universal.system.JobError as e:
            logger.error(e)



def check_permissions(model_names=None):
    logger.info('Checking permissions.')
    
    base_dirs = get_base_dirs(model_names=model_names)
    
    def check_file(file):
        permissions = os.stat(file)[stat.ST_MODE]
        if not (permissions & stat.S_IRUSR and permissions & stat.S_IRGRP):
            logger.error('File {} is not readable!'.format(file))
    def check_dir(file):
        permissions = os.stat(file)[stat.ST_MODE]
        if not (permissions & stat.S_IRUSR and permissions & stat.S_IXUSR and permissions & stat.S_IRGRP and permissions & stat.S_IXGRP):
            logger.error('Dir {} is not readable!'.format(file))

    for base_dir in base_dirs:
        util.io.fs.walk_all_in_dir(base_dir, check_file, check_dir, exclude_dir=False, topdown=True)



def check_db(model_names=None):
    logger.info('Checking database integrity.')
    
    model = simulation.model.cache.Model()
    try:
        model.check_integrity()
    except (util.index_database.general.DatabaseError, simulation.model.eval.DatabaseError, OSError) as e:
        logger.error(e)



if __name__ == "__main__":
    ## configure arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--check_job_options', action='store_true')
    parser.add_argument('-p', '--check_permissions', action='store_true')
    parser.add_argument('-d', '--check_database', action='store_true')
    parser.add_argument('-m', '--model', default=None, help='The model to check. If not specified all models are checked')
    args = parser.parse_args()
    ## run check
    with util.logging.Logger():
        if args.check_job_options:
            check_job_options(model_names=args.model)
        if args.check_permissions:
            check_permissions(model_names=args.model)
        if args.check_database:
            check_db(model_names=args.model)
        logger.info('Check completed.')

