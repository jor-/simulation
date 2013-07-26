import scipy.io

from ndop.optimization.cost_function import Cost_function

from ndop.optimization.matlab_eval_constants import MATLAB_EVAL_YEARS, MATLAB_EVAL_TOLERANCE, MATLAB_EVAL_TIME_STEP_SIZE, MATLAB_EVAL_DEBUG_LEVEL, MATLAB_EVAL_PARAMETER_FILE, MATLAB_EVAL_DF_FILE

cf = Cost_function(years=MATLAB_EVAL_YEARS, tolerance=MATLAB_EVAL_TOLERANCE, time_step_size=MATLAB_EVAL_TIME_STEP_SIZE, debug_level=MATLAB_EVAL_DEBUG_LEVEL)

p = scipy.io.loadmat(MATLAB_EVAL_PARAMETER_FILE, squeeze_me=True)
p = p['p']

df = {}
df['df'] = cf.df(p)
scipy.io.savemat(MATLAB_EVAL_DF_FILE, df, oned_as='column')