import numpy as np

import util.batch.universal.system

from ndop.model.constants import JOB_MEMORY_GB


COST_FUNCTION_DIRNAME = 'cost_functions'
COST_FUNCTION_F_FILENAME = 'f.npy'
COST_FUNCTION_DF_FILENAME = 'df.npy'
COST_FUNCTION_F_NORMALIZED_FILENAME = 'f_normalized.npy'
COST_FUNCTION_GLS_PROD_FILENAME = 'inv_cov_matrix_mult_residuum.npy'
COST_FUNCTION_CORRELATION_PARAMETER_FILENAME = 'cp.npy'
COST_FUNCTION_F_OPTION_FILENAME = 'f_options.npy'
COST_FUNCTION_DF_OPTION_FILENAME = 'df_options.npy'

CONCENTRATION_MIN_VALUE = 10**(-6)

# COST_FUNCTION_NODES_SETUP_SPINUP = util.batch.universal.system.NodeSetup(memory=JOB_MEMORY_GB, node_kind='f_ocean2', nodes=6, cpus=16)
# COST_FUNCTION_NODES_SETUP_DERIVATIVE = util.batch.universal.system.NodeSetup(memory=JOB_MEMORY_GB, node_kind=('f_ocean2', 'f_ocean'), total_cpus_min=48)
# COST_FUNCTION_NODES_SETUP_TRAJECTORY = util.batch.universal.system.NodeSetup(memory=JOB_MEMORY_GB, node_kind=('f_ocean', 'westmere', 'shanghai', 'f_ocean2'), nodes=1, total_cpus_min=4)
# COST_FUNCTION_NODES_SETUP_JOB = util.batch.universal.system.NodeSetup(node_kind=('f_ocean', 'westmere', 'shanghai'), nodes=1, cpus=1)


if util.batch.universal.system.IS_RZ:
    COST_FUNCTION_NODES_SETUP_SPINUP = util.batch.universal.system.NodeSetup(memory=JOB_MEMORY_GB, node_kind='f_ocean2', nodes=6, cpus=16)
    COST_FUNCTION_NODES_SETUP_DERIVATIVE = util.batch.universal.system.NodeSetup(memory=JOB_MEMORY_GB, node_kind=('f_ocean2', 'f_ocean'), total_cpus_min=48)
    COST_FUNCTION_NODES_SETUP_TRAJECTORY = util.batch.universal.system.NodeSetup(memory=JOB_MEMORY_GB, node_kind=('f_ocean', 'westmere', 'shanghai', 'f_ocean2'), nodes=1, total_cpus_min=4)
    COST_FUNCTION_NODES_SETUP_JOB = util.batch.universal.system.NodeSetup(node_kind=('f_ocean', 'westmere', 'shanghai'), nodes=1, cpus=1)

if util.batch.universal.system.IS_NEC:
    # COST_FUNCTION_NODES_SETUP_SPINUP = util.batch.universal.system.NodeSetup(memory=JOB_MEMORY_GB, node_kind='clmedium', nodes=10, cpus=16)
    # COST_FUNCTION_NODES_SETUP_DERIVATIVE = util.batch.universal.system.NodeSetup(memory=JOB_MEMORY_GB, node_kind='clmedium', nodes=5, cpus=16)
    COST_FUNCTION_NODES_SETUP_SPINUP = util.batch.universal.system.NodeSetup(memory=JOB_MEMORY_GB, node_kind='clfo2', nodes=6, cpus=24)
    COST_FUNCTION_NODES_SETUP_DERIVATIVE = util.batch.universal.system.NodeSetup(memory=JOB_MEMORY_GB, node_kind='clmedium', nodes=4, cpus=16)
    COST_FUNCTION_NODES_SETUP_TRAJECTORY = util.batch.universal.system.NodeSetup(memory=JOB_MEMORY_GB, node_kind='clexpress', nodes=1, cpus=16)
    COST_FUNCTION_NODES_SETUP_JOB = util.batch.universal.system.NodeSetup(node_kind='clmedium', nodes=1, cpus=1)
    
    

