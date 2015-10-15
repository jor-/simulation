MATLAB_PARAMETER_FILENAME = 'p.mat'
MATLAB_F_FILENAME = 'f.mat'
MATLAB_DF_FILENAME = 'df.mat'
NODES_MAX_FILENAME = 'max_nodes.txt'

GLS_DICT = {'WOD': (35, 40), 'WOD.0': (20, 25, 30, 35, 40), 'WOD.1': (25, 30, 35, 40)}
KIND_OF_COST_FUNCTIONS = ['{}_{}'.format(dk, cf) for dk in ('WOA', 'WOD', 'WOD.1', 'WOD.0') for cf in ('OLS', 'WLS', 'LWLS')] + ['{}_GLS.{}.-1'.format(dk, mv) for dk in GLS_DICT.keys() for mv in GLS_DICT[dk]]
