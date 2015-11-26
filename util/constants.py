import os.path

OPTION_FILE_SUFFIX = '_options'

CACHE_DIRNAME = 'output'

BOXES_CACHE_DIRNAME = 'boxes'
BOXES_F_FILENAME = os.path.join(BOXES_CACHE_DIRNAME, 'f.npy')
BOXES_DF_FILENAME = os.path.join(BOXES_CACHE_DIRNAME, 'df_-_step_size_{step_size:g}.npy')

WOA_CACHE_DIRNAME = 'WOA'
WOA_F_FILENAME = os.path.join(WOA_CACHE_DIRNAME, 'f_pw.npy')
WOA_DF_FILENAME = os.path.join(WOA_CACHE_DIRNAME, 'df_pw_-_step_size_{step_size:g}.npy')

WOD_CACHE_DIRNAME = 'WOD'
WOD_F_FILENAME = os.path.join(WOD_CACHE_DIRNAME, 'f_pw_lexsorted.npy')
WOD_DF_FILENAME = os.path.join(WOD_CACHE_DIRNAME, 'df_pw_lexsorted_-_step_size_{step_size:g}.npy')

