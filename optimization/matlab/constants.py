MATLAB_PARAMETER_FILENAME = 'p.mat'
MATLAB_F_FILENAME = 'f.mat'
MATLAB_DF_FILENAME = 'df.mat'
NODES_MAX_FILENAME = 'max_nodes.txt'

KIND_OF_COST_FUNCTIONS = ['{}_{}'.format(dk, cf) for dk in ('WOA', 'WOD', 'WOD.1', 'WOD.0') for cf in ('OLS', 'WLS', 'LWLS')] + ['{}_GLS.{}.-1'.format(dk, mv) for dk in ('WOD', 'WOD.1', 'WOD.0') for mv in (30, 35, 40)] + ['{}_GLS.{}.-1'.format(dk, mv) for dk in ('WOD.0',) for mv in (20, 25)]
