import numpy as np

import util.rzcluster.interact

from ndop.model.constants import JOB_MEMORY_GB


COST_FUNCTIONS_DIRNAME = 'cost_functions'
COST_FUNCTION_F_FILENAME = 'f.npy'
COST_FUNCTION_DF_FILENAME = 'df.npy'
COST_FUNCTION_F_NORMALIZED_FILENAME = 'f_normalized.npy'
COST_FUNCTION_GLS_PROD_FILENAME = 'inv_cov_matrix_mult_residuum.npy'
COST_FUNCTION_CORRELATION_PARAMETER_FILENAME = 'cp.npy'
COST_FUNCTION_F_OPTION_FILENAME = 'f_options.npy'
COST_FUNCTION_DF_OPTION_FILENAME = 'df_options.npy'

# # COST_FUNCTION_NODES_SETUP_SPINUP = ('f_ocean2', 5, 16)
# # COST_FUNCTION_NODES_SETUP_DERIVATIVE = ('foexpress', 2, 16)
# # COST_FUNCTION_NODES_SETUP_TRAJECTORY = ('foexpress', 1, 16)
# COST_FUNCTION_NODES_SETUP_SPINUP = ('f_ocean2', 6, 16)
# # COST_FUNCTION_NODES_SETUP_DERIVATIVE = ('f_ocean2', 6, 16)
# # COST_FUNCTION_NODES_SETUP_DERIVATIVE = ('f_ocean', 8, 8)
# COST_FUNCTION_NODES_SETUP_DERIVATIVE = ('f_ocean',)
# # COST_FUNCTION_NODES_SETUP_TRAJECTORY = ('westmere', 1, 1)
# COST_FUNCTION_NODES_SETUP_TRAJECTORY = ('f_ocean', 1, 8)

# COST_FUNCTION_NODES_SETUP_SPINUP = util.rzcluster.interact.NodeSetup(memory=JOB_MEMORY_GB, node_kind='f_ocean2', total_cpus_min=48, nodes_max=6)
# COST_FUNCTION_NODES_SETUP_DERIVATIVE = util.rzcluster.interact.NodeSetup(memory=JOB_MEMORY_GB, node_kind='f_ocean', total_cpus_min=48)
# COST_FUNCTION_NODES_SETUP_TRAJECTORY = util.rzcluster.interact.NodeSetup(memory=JOB_MEMORY_GB, node_kind=('westmere', 'shanghai', 'f_ocean'), total_cpus_min=4, nodes_max=1)
COST_FUNCTION_NODES_SETUP_SPINUP = util.rzcluster.interact.NodeSetup(memory=JOB_MEMORY_GB, node_kind='f_ocean2', nodes=6, cpus=16)
COST_FUNCTION_NODES_SETUP_DERIVATIVE = util.rzcluster.interact.NodeSetup(memory=JOB_MEMORY_GB, node_kind=('f_ocean2', 'f_ocean'), total_cpus_min=48)
# COST_FUNCTION_NODES_SETUP_TRAJECTORY = util.rzcluster.interact.NodeSetup(memory=JOB_MEMORY_GB, node_kind=('f_ocean2', 'f_ocean'), total_cpus_min=4, nodes_max=1)
# COST_FUNCTION_NODES_SETUP_DERIVATIVE = util.rzcluster.interact.NodeSetup(memory=JOB_MEMORY_GB, node_kind='f_ocean2', nodes=6, cpus=16)
COST_FUNCTION_NODES_SETUP_TRAJECTORY = util.rzcluster.interact.NodeSetup(memory=JOB_MEMORY_GB, node_kind=('f_ocean', 'westmere', 'shanghai'), nodes=1, total_cpus_min=4)
COST_FUNCTION_NODES_SETUP_JOB = util.rzcluster.interact.NodeSetup(node_kind=('f_ocean', 'westmere', 'shanghai'), nodes=1, cpus=1)
# COST_FUNCTION_NODES_SETUP_JOB = util.rzcluster.interact.NodeSetup(node_kind='shanghai', nodes=1, cpus=1)

# COST_FUNCTION_JOB_NODE_KIND = 'f_ocean2'




# PARAMETER_BOUNDS = np.array([[0.05, 0.95], [0.5, 10], [0.05, 0.95], [0.005, 10], [10, 50], [0.001, 0.2], [0.7 , 1.3]]).T
CONCENTRATION_MIN_VALUE = 10**(-6)

