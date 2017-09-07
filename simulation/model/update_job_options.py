import os
import stat

import numpy as np

import simulation.constants
import simulation.model.job
import simulation.model.constants

import util.io.fs
import util.options
import util.petsc.universal
import util.logging


# general update functions for job options

def update_job_options(update_function, model_names=None):
    if model_names is None:
        database_dir = simulation.model.constants.DATABASE_OUTPUT_DIR
        util.logging.info('Getting jobs in {}.'.format(database_dir))
        job_files = util.io.fs.get_files(database_dir, filename_pattern='*/job_options.hdf5', use_absolute_filenames=True, recursive=True)
        util.logging.info('Got {} jobs.'.format(len(job_files)))
    else:
        job_files = []
        for model_name in model_names:
            model_dirname = simulation.model.constants.DATABASE_MODEL_DIRNAME.format(model_name)
            model_dir = os.path.join(simulation.model.constants.DATABASE_OUTPUT_DIR, model_dirname)
            util.logging.info('Getting jobs in {}.'.format(model_dir))
            model_job_files = util.io.fs.get_files(model_dir, filename_pattern='*/job_options.hdf5', use_absolute_filenames=True, recursive=True)
            util.logging.info('Got {} jobs.'.format(len(model_job_files)))
            job_files.extend(model_job_files)


    for job_file in job_files:
        util.io.fs.make_writable(job_file)
        update_function(job_file)
        util.io.fs.make_read_only(job_file)



# specific update functions for job options

def update_function_output_dir(job_file):
    with util.options.OptionsFile(job_file) as options:
        old_output_dir = options['/metos3d/output_dir']
        if old_output_dir.endswith('/'):
            old_output_dir = old_output_dir[:-1]

        new_output_dir = os.path.dirname(job_file)
        new_output_dir = new_output_dir.replace(simulation.constants.SIMULATION_OUTPUT_DIR, '${{{}}}'.format((simulation.constants.SIMULATION_OUTPUT_DIR_ENV_NAME)))

        if new_output_dir != old_output_dir:
            util.logging.info('Changing output path from {} to {}.'.format(old_output_dir, new_output_dir))
            options.replace_all_str_options(old_output_dir, new_output_dir)



#def update_function_str_options(job_file):
#    with util.options.OptionsFile(job_file) as options:
#        options.replace_all_str_options(old_str, new_str)



#def update_function_option_entries(job_file):
#    with util.options.OptionsFile(job_file) as options:

        # # change constant_concentrations name
        # try:
        #     options['/model/initial_concentrations']
        # except KeyError:
        #     options['/model/initial_constant_concentrations'] = np.array([2.17, 10**-4])
        #     util.logging.info('Setting default value for /model/initial_constant_concentrations in job option file {}.'.format(job_file))
        # else:
        #     options['/model/initial_constant_concentrations'] = options['/model/initial_concentrations']
        #     util.logging.info('Renaming /model/initial_concentrations to /model/initial_constant_concentrations in job option file {}.'.format(job_file))
        #     del options['/model/initial_concentrations']
        #
        # # remove concentration factor
        # try:
        #     options['/model/total_concentration_factor']
        # except KeyError:
        #     pass
        # else:
        #     concentration = options['/model/total_concentration_factor'] * np.array([2.17, 10**-4])
        #     saved_concentration = options['/model/initial_constant_concentrations']
        #
        #     if np.any(saved_concentration != concentration):
        #         util.logging.info('Correcting /model/initial_constant_concentrations with /model/total_concentration_factor in job option file {}.'.format(job_file))
        #         options['/model/initial_constant_concentrations'] = concentration

        #       del options['/model/total_concentration_factor']
        #     util.logging.info('Deleting /model/total_concentration_factor in job option file {}.'.format(job_file))
        #
        # # set model name
        # options['/model/name'] = 'MITgcm-PO4-DOP'
        # util.logging.info('Setting /model/name in job option file {}.'.format(job_file))
        #
        # # '/model/tracer_input_dir' -> '/model/tracer_input_files'
        #
        # try:
        #     tracer_input_dir = options['/model/tracer_input_dir']
        # except KeyError:
        #     pass
        # else:
        #     tracer_input_files = [os.path.join(tracer_input_dir, input_file) for input_file in ['dop_input.petsc', 'po4_input.petsc']]
        #     options['/model/tracer_input_files'] = tracer_input_files
        #     del options['/model/tracer_input_dir']
        #     util.logging.info('Setting /model/tracer_input_files in job option file {}.'.format(job_file))
        #
        #
        # # replace dirs with env
        # options.replace_all_str_options('/sfs/fs3/work-sh1/sunip229/metos3d', '${METOS3D_DIR}')
        # options.replace_all_str_options('${METOS3D_DIR}/data/Metos3DData', '${METOS3D_DIR}/data/data/TMM/2.8')


        #
        # try:
        #     options['/metos3d/tracer_input_path']
        #     options['/metos3d/initial_concentrations']
        # except KeyError:
        #     pass
        # else:
        #     del options['/metos3d/initial_concentrations']
        #     util.logging.info('Initial concentration option removed, since tracer input is available to job option file {}.'.format(job_file))
        #
        # try:
        #     options['/model/concentrations']
        # except KeyError:
        #     pass
        # else:
        #     del options['/model/concentrations']
        #     util.logging.info('Concentrations option removed in job option file {}.'.format(job_file))
        #
        # try:
        #     options['/model/initial_concentrations']
        # except KeyError:
        #     try:
        #         options['/metos3d/tracer_input_path']
        #     except KeyError:
        #         options['/model/initial_concentrations'] = np.array([2.17, 10**-4])
        #         util.logging.info('Model initial concentration option added to job option file {}.'.format(job_file))
        #
        # try:
        #     options['/metos3d/initial_concentrations']
        # except KeyError:
        #     pass
        # else:
        #     del options['/metos3d/initial_concentrations']
        #     util.logging.info('Metos3d concentrations option removed in job option file {}.'.format(job_file))
        #
        # try:
        #     options['/model/time_step_multiplier']
        # except KeyError:
        #     options['/model/time_step_multiplier'] = 1
        #     util.logging.info('time_step_multiplier option added to job option file {}.'.format(job_file))
        #
        # try:
        #     options['/model/time_steps_per_year']
        # except KeyError:
        #     options['/model/time_steps_per_year'] = options['/model/time_step_count']
        #     del options['/model/time_step_count']
        #     util.logging.info('time_steps_per_year option added to job option file {}.'.format(job_file))
        #
        # try:
        #     options['/model/tracer']
        # except KeyError:
        #     options['/model/tracer'] = ['po4', 'dop']
        #     util.logging.info('model tracer option added to job option file {}.'.format(job_file))
        #
        # try:
        #     options['/metos3d/po4_output_filename']
        # except KeyError:
        #     pass
        # else:
        #     del options['/metos3d/po4_output_filename']
        #     del options['/metos3d/dop_output_filename']
        #     options['/metos3d/output_filenames'] = ['{}_output.petsc'.format(tracer) for tracer in options['/model/tracer']]
        #     util.logging.info('generic output filenames added to job option file {}.'.format(job_file))
        #
        # try:
        #     options['/metos3d/po4_input_filename']
        # except KeyError:
        #     pass
        # else:
        #     del options['/metos3d/po4_input_filename']
        #     del options['/metos3d/dop_input_filename']
        #     options['/metos3d/input_filenames'] = ['{}_input.petsc'.format(tracer) for tracer in options['/model/tracer']]
        #     util.logging.info('generic input filenames added to job option file {}.'.format(job_file))
        #
        # try:
        #     options['/metos3d/tracer_input_path']
        # except KeyError:
        #     pass
        # else:
        #     del options['/metos3d/tracer_input_path']
        #     util.logging.info('Metos3d tracer input path removed from job option file {}.'.format(job_file))
        #
        # try:
        #     options['/metos3d/path']
        # except KeyError:
        #     pass
        # else:
        #     del options['/metos3d/path']
        #     util.logging.info('Metos3d path removed from job option file {}.'.format(job_file))
        #
        # try:
        #     options['/model/tracer_input_path']
        # except KeyError:
        #     pass
        # else:
        #     del options['/model/tracer_input_path']
        #     util.logging.info('Model tracer input path removed from job option file {}.'.format(job_file))
        #
        # try:
        #     options['/metos3d/output_dir']
        # except KeyError:
        #     options['/metos3d/output_dir'] = options['/metos3d/output_dir']
        #     del options['/metos3d/output_dir']
        #     util.logging.info('Metos3d output_dir remamed to ouput_dir in job option file {}.'.format(job_file))
        #
        #   try:
        #     options['/metos3d/tracer_output_dir']
        # except KeyError:
        #     options['/metos3d/tracer_output_dir'] = options['/metos3d/output_dir']
        #     util.logging.info('Metos3d tracer output dir added to job option file {}.'.format(job_file))
        #
        #
        # try:
        #     input_tracer_filename = options['/metos3d/input_filenames'][0]
        # except KeyError:
        #     input_tracer_filename = None
        # if input_tracer_filename is not None:
        #     input_tracer_dir = options['/metos3d/output_dir'].replace('${{{}}}'.format(simulation.constants.SIMULATION_OUTPUT_DIR_ENV_NAME), simulation.constants.SIMULATION_OUTPUT_DIR)
        #     input_tracer = os.path.join(input_tracer_dir, input_tracer_filename)
        #
        #     correct_metos3d_tracer_input_dir = options['/metos3d/output_dir']
        #     try:
        #         options['/metos3d/tracer_input_dir']
        #     except KeyError:
        #         options['/metos3d/tracer_input_dir'] = correct_metos3d_tracer_input_dir
        #         util.logging.info('Metos3d tracer input dir was not set, added to job option file {}.'.format(job_file))
        #
        #       correct_model_tracer_input_dir = os.path.dirname(os.path.realpath(input_tracer)).replace(simulation.constants.SIMULATION_OUTPUT_DIR, '${{{}}}'.format(simulation.constants.SIMULATION_OUTPUT_DIR_ENV_NAME))
        #     try:
        #         options['/model/tracer_input_dir']
        #     except KeyError:
        #         options['/model/tracer_input_dir'] = correct_model_tracer_input_dir
        #         util.logging.info('Model tracer input dir added to job option file {}.'.format(job_file))

        #   try:
        #     options['/model/parameters_file']
        # except KeyError:
        #     pass
        # else:
        #     del options['/model/parameters_file']
        #     util.logging.info('Model parameters file removed from job option file {}.'.format(job_file))
        #
        # try:
        #     options['/metos3d/data_path']
        # except KeyError:
        #     pass
        # else:
        #     options['/metos3d/data_dir'] = options['/metos3d/data_path']
        #     del options['/metos3d/data_path']
        #     util.logging.info('/metos3d/data_path renamed to /metos3d/data_dir in job option file {}.'.format(job_file))

        #   try:
        #     options['/metos3d/data_path']
        # except KeyError:
        #     pass
        # else:
        #     options['/metos3d/data_dir'] = options['/metos3d/data_path']
        #     del options['/metos3d/data_path']
        #     util.logging.info('/metos3d/data_path renamed to /metos3d/data_dir in job option file {}.'.format(job_file))

        #   try:
        #     options['/metos3d/tracer_output_dir']
        # except KeyError:
        #     pass
        # else:
        #     options['/metos3d/tracer_output_dir'] = options['/metos3d/tracer_output_dir']
        #     del options['/metos3d/tracer_output_dir']
        #     util.logging.info('/metos3d/tracer_output_dir renamed to /metos3d/tracer_output_dir in job option file {}.'.format(job_file))

        #   try:
        #     options['/metos3d/output_dir']
        # except KeyError:
        #     pass
        # else:
        #     options['/metos3d/output_dir'] = options['/metos3d/output_dir']
        #     del options['/metos3d/output_dir']
        #     util.logging.info('/metos3d/output_dir renamed to /metos3d/output_dir in job option file {}.'.format(job_file))

        #   try:
        #     options['/job/unfinished_file']
        # except KeyError:
        #     options['/job/unfinished_file'] = os.path.join(job_options_dir, 'unfinished.txt')
        #     util.logging.info('/job/unfinished_file added to job option file {}.'.format(job_file))
        #
        # try:
        #     options['/metos3d/tolerance']
        # except KeyError:
        #     pass
        # else:
        #     options['/model/spinup/tolerance'] = options['/metos3d/tolerance']
        #     del options['/metos3d/tolerance']
        #     util.logging.info('/metos3d/tolerance renamed to /model/spinup/tolerance in job option file {}.'.format(job_file))
        #
        # try:
        #     options['/metos3d/years']
        # except KeyError:
        #     pass
        # else:
        #     options['/model/spinup/years'] = options['/metos3d/years']
        #     del options['/metos3d/years']
        #     util.logging.info('/metos3d/years renamed to /model/spinup/years in job option file {}.'.format(job_file))

#        try:
#            options['/model/tracer_input_files']
#        except KeyError:
#            pass
#        else:
#            files = options['/model/tracer_input_files']
#            output_dir = options['/metos3d/output_dir']
#
#            # replace
#            def remove_to_parameter_set(value):
#                while not os.path.basename(value).startswith('parameter_set'):
#                    assert len(value) > 0
#                    value = os.path.dirname(value)
#                return value
#
#            parameter_set_base_dir_correct = remove_to_parameter_set(output_dir)
#
#            def replace_tracer_input_file(file):
#                file = file.replace('dop_input.petsc', 'dop_output.petsc')
#                file = file.replace('po4_input.petsc', 'po4_output.petsc')
#
#                parameter_set_base_dir_wrong = remove_to_parameter_set(file)
#                file = parameter_set_base_dir_correct + file[len(parameter_set_base_dir_wrong):]
#                return file
#
#            new_files = tuple(map(replace_tracer_input_file, files))
#
#            if any(['derivative' in new_file for new_file in new_files]):
#                spinup_dir = os.path.join(parameter_set_base_dir_correct, 'spinup')
#                run_dirs = os.listdir(path=os.path.expandvars(spinup_dir))
#                assert len(run_dirs) == 1
#                new_files = tuple(os.path.join(spinup_dir, run_dirs[0], tracer_input_filename) for tracer_input_filename in options['/metos3d/output_filenames'])
#
#            assert len(new_files) == 2
#
#            # correct order
#            if 'po4' in new_files[1]:
#                new_files = new_files[::-1]
#
#            # check if exist
#            new_files_expanded = tuple(map(os.path.expandvars, new_files))
#            assert all(map(os.path.exists, new_files_expanded))
#
#            # check if same
#            metos_tracer_input_files = (os.path.join(output_dir, tracer_input_filename) for tracer_input_filename in options['/metos3d/input_filenames'])
#            metos_tracer_input_files_expanded = tuple(map(os.path.expandvars, metos_tracer_input_files))
#            assert all((np.allclose(util.petsc.universal.load_petsc_vec_to_numpy_array(a), util.petsc.universal.load_petsc_vec_to_numpy_array(b)) for a, b in zip(new_files_expanded, metos_tracer_input_files_expanded)))
#
#            # set
#            if any(options['/model/tracer_input_files'] != new_files):
#                util.logging.info('/model/tracer_input_files: {} replaced by {}.'.format(files, new_files))
#                options['/model/tracer_input_files'] = new_files
#        except AssertionError:
#            util.logging.error('Could not update /model/tracer_input_files: {}.'.format(files))





def update_run_dirs_in_job_options():
    def update_function(job_file):
        with util.options.OptionsFile(job_file) as options:
            for i in range(9):
                old_str = 'run_{:0>2d}/'.format(i)
                new_str = 'run_{:0>5d}/'.format(i)
                options.replace_all_str_options(old_str, new_str)
    update_job_options(update_function)



def update_tracer_input_files_in_job_options():
    def update_function(job_file):
        with util.options.OptionsFile(job_file) as options:
            try:
                options['/model/tracer_input_files']
            except KeyError:
                pass
            else:
                metos_tracer_input_dir = options['/metos3d/tracer_input_dir']

                if 'derivative' in metos_tracer_input_dir:
                    base_dir = metos_tracer_input_dir[:metos_tracer_input_dir.find('derivative')]
                else:
                    assert 'spinup' in metos_tracer_input_dir
                    base_dir = metos_tracer_input_dir[:metos_tracer_input_dir.find('spinup')]

                model_tracer_input_files_old = options['/model/tracer_input_files']

                model_tracer_input_files_new = tuple(base_dir + model_tracer_input_file[model_tracer_input_file.find('spinup'):] for model_tracer_input_file in model_tracer_input_files_old)
                model_tracer_input_files_new = tuple(model_tracer_input_file.replace('input.petsc', 'output.petsc') for model_tracer_input_file in model_tracer_input_files_new)

                options['/model/tracer_input_files'] = model_tracer_input_files_new

                util.logging.info('Changing "/metos3d/tracer_input_dir" from {} to {}.'.format(model_tracer_input_files_old, model_tracer_input_files_new))
    update_job_options(update_function)



# general update functions for parameter files

def update_parameter_files(update_function):
    from simulation.model.constants import MODEL_NAMES, DATABASE_OUTPUT_DIR, DATABASE_MODEL_DIRNAME, DATABASE_PARAMETERS_FILENAME

    for model_name in MODEL_NAMES:
        model_dirname = DATABASE_MODEL_DIRNAME.format(model_name)
        model_dir = os.path.join(DATABASE_OUTPUT_DIR, model_dirname)

        for time_step_dir in util.io.fs.get_dirs(model_dir, use_absolute_filenames=True):
            parameter_set_dirs = util.io.fs.get_dirs(time_step_dir, use_absolute_filenames=True)
            util.logging.debug('{} parameter set dirs found in {}.'.format(len(parameter_set_dirs), time_step_dir))

            for parameter_set_dir in parameter_set_dirs:
                parameter_file = os.path.join(parameter_set_dir, DATABASE_PARAMETERS_FILENAME)

                util.io.fs.make_writable(parameter_file)
                p = np.loadtxt(parameter_file)
                p = update_function(p)
                np.savetxt(parameter_file, p)
                util.io.fs.make_read_only(parameter_file)



if __name__ == "__main__":
    with util.logging.Logger():
        # update_str_options('$NDOP_DIR/model_output', '${SIMULATION_OUTPUT_DIR}/model_dop_po4')
        # update_str_options('${NDOP_DIR}/model_output', '${SIMULATION_OUTPUT_DIR}/model_dop_po4')
        # update_str_options('${MODEL_OUTPUT_DIR}/time_step_0001', '${SIMULATION_OUTPUT_DIR}/model_dop_po4/time_step_0001')
        # model_names = ['MITgcm-PO4-DOP']
        # update_new_option_entries(model_names=model_names)
        # update_output_dir(model_names=model_names)
        # update_parameter_files_add_total_concentration_factors()
        # update_run_dirs_in_job_options()
        # update_str_options('model_dop_po4', 'model_MITgcm-PO4-DOP')
        # update_tracer_input_files_in_job_options()
        # def update_function (job_file):
        #     update_function_output_dir(job_file);
        #     update_function_option_entries(job_file)
#        update_job_options(update_function_option_entries, model_names=['MITgcm-PO4-DOP'])
        update_job_options(update_function_output_dir)

    util.logging.info('Update completed.')
