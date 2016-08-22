MATLAB_PARAMETER_FILENAME = 'p.mat'
MATLAB_F_FILENAME = 'f.mat'
MATLAB_DF_FILENAME = 'df.mat'
NODES_MAX_FILENAME = 'max_nodes.txt'

# DATA_KINDS = ('WOA', 'WOD', 'WOD.1', 'WOD.0', 'OLDWOD.1')
# GLS_DICT = {'WOD': (25, 30, 35, 40, 45, 50), 'WOD.0': (20, 25, 30, 35, 40, 45, 50), 'WOD.1': (25, 30, 35, 40, 45, 50), 'OLDWOD.1': (25, 30, 35, 40), 'WOA': ()}
# COST_FUNCTION_NAMES = ['{}_{}'.format(dk, cf) for dk in DATA_KINDS for cf in ('OLS', 'WLS', 'LWLS')] + ['{}_GLS.{}.-1'.format(dk, mv) for dk in DATA_KINDS for mv in GLS_DICT[dk]]
COST_FUNCTION_NAMES = ('OLS', 'WLS', 'LWLS', 'GLS')

