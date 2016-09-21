import os
import shutil

import numpy as np

import simulation.model.eval
import util.io.fs


def update_db():
    BASE_DIR = '$SIMULATION_OUTPUT_DIR/model_dop_po4/time_step_0001'
    DB_FILE = os.path.join(BASE_DIR, 'database.npy')
    db_array = np.load(DB_FILE)
    PARAMETER_SET_DIR = os.path.join(BASE_DIR, 'parameter_set_{:0>5d}')
    BASE_CONCENTRATION = np.array([2.17, 10**-4])
    
    model_db = simulation.model.eval.Model_Database()
    model_options = model_db.model_options
    
    for i in range(len(db_array)):
        p_old = db_array[i]
        assert len(p_old) == 8
        
        if not np.any(np.isnan(p_old)):
            p_new = p_old[:-1]
            assert len(p_new) == 7
            concentration_factor = p_old[-1]
            concentration = BASE_CONCENTRATION * concentration_factor
            parameter_set_dir_old = PARAMETER_SET_DIR.format(i)
            
            try:
                model_options.parameters = p_new
            except ValueError:
                print('Parameter {} are not allowed!'.format(p_new))
            else:
                model_options.initial_concentration_options.constant_concentrations = concentration
                parameter_set_dir_new = model_db.parameter_set_dir
                
                for dir in util.io.fs.get_dirs(parameter_set_dir_old, use_absolute_filenames=False):
                    abs_dir_from = os.path.join(parameter_set_dir_old, dir)
                    abs_dir_to = os.path.join(parameter_set_dir_new, dir)
                    print('Coping {} to {}.'.format(abs_dir_from, abs_dir_to))
                    shutil.copytree(abs_dir_from, abs_dir_to, symlinks=True)

update_db()