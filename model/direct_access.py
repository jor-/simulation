import os.path
import numpy as np

import util.pattern

import logging
logger = logging.getLogger(__name__)


def f(parameter_set_dir, t_dim=12):
    from ndop.model.constants import MODEL_F_FILENAME
    
    f_filepattern = os.path.join(parameter_set_dir, MODEL_F_FILENAME)
    f_file = f_filepattern.format(t_dim)
    
    logger.debug('Try to load F from %s.' % f_file)
    try:
        f = np.load(f_file, 'r')
        logger.debug('F file loaded from %s.' % f_file)
    except (OSError, IOError) as e:
        logger.debug('No F file found.')
        raise e
    
    return f



def df(parameter_set_dir, t_dim=12):
    from ndop.model.constants import MODEL_DF_FILENAME
    
    df = None
    accuracy_order = 2
    
    while df is None and accuracy_order > 0:
        df_filepattern = os.path.join(parameter_set_dir, MODEL_DF_FILENAME)
        df_file = df_filepattern.format(t_dim, accuracy_order)
        
        logger.debug('Try to load DF from %s.' % df_file)
        try:
            df = np.load(df_file, 'r')
            logger.debug('DF File loaded from %s.' % df_file)
        except (OSError, IOError) as e:
            if accuracy_order > 1:
                logger.debug('Tried to load DF file from %s but it does not exists.' % df_file)
                accuracy_order -= 1
            else:
                logger.debug('No DF file found.')
                raise e
    
    return df
