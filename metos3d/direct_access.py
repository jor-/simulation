import os.path
import numpy as np

import util.pattern
from util.debug import print_debug

def get_df(parameter_set_dir, t_dim =  12, debug_level=0, required_debug_level=1):
    from ndop.metos3d.constants import MODEL_DF_FILENAME
    BASENAME = 'ndop.metos3d.direct_access.get_df' 
    
    df = None
    accuracy_order = 2
    
    while df is None and accuracy_order > 0:
        df_filepattern = os.path.join(parameter_set_dir, MODEL_DF_FILENAME)
        df_file = util.pattern.replace_int_pattern(df_filepattern, (t_dim, accuracy_order))
        
        print_debug(('Try to load DF from "', df_file, '".'), debug_level, required_debug_level, BASENAME)
        try:
            df = np.load(df_file, 'r')
            print_debug(('DF File loaded from "', df_file, '".'), debug_level, required_debug_level, BASENAME)
        except (OSError, IOError) as e:
            if accuracy_order > 1:
                print_debug(('Tried to load DF file from "', df_file, '" but it does not exists.'), debug_level, required_debug_level, BASENAME)
                accuracy_order -= 1
            else:
                print_debug('No DF file found.', debug_level, required_debug_level, BASENAME)
                raise e
    
    return df



def get_f(parameter_set_dir, t_dim =  12, debug_level=0, required_debug_level=1):
    from ndop.metos3d.constants import MODEL_F_FILENAME
    BASENAME = 'ndop.metos3d.direct_access.get_f' 
    
    f_filepattern = os.path.join(parameter_set_dir, MODEL_F_FILENAME)
    f_file = util.pattern.replace_int_pattern(f_filepattern, t_dim)
    
    print_debug(('Try to load F from "', f_file, '".'), debug_level, required_debug_level, BASENAME)
    try:
        f = np.load(f_file, 'r')
        print_debug(('F file loaded from "', f_file, '".'), debug_level, required_debug_level, BASENAME)
    except (OSError, IOError) as e:
        print_debug('No F file found.', debug_level, required_debug_level, BASENAME)
        raise e
    
    return f