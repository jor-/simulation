import numpy as np

import util.batch.universal.system

from ndop.model.constants import JOB_MEMORY_GB

COST_FUNCTION_DIRNAME = 'cost_functions'
COST_FUNCTION_F_FILENAME = 'f.npy'
COST_FUNCTION_DF_FILENAME = 'df_-_step_size_{step_size:g}.npy'
COST_FUNCTION_F_NORMALIZED_FILENAME = 'f_normalized.npy'
COST_FUNCTION_GLS_PROD_FILENAME = 'inv_cov_matrix_mult_residuum.npy'
COST_FUNCTION_CORRELATION_PARAMETER_FILENAME = 'cp.npy'

CONCENTRATION_MIN_VALUE = 10**(-6)


if util.batch.universal.system.IS_RZ:
    COST_FUNCTION_NODES_SETUP_SPINUP = util.batch.universal.system.NodeSetup(memory=JOB_MEMORY_GB, node_kind='f_ocean2', nodes=6, cpus=16, total_cpus_max=9*16, check_for_better=True)
    COST_FUNCTION_NODES_SETUP_DERIVATIVE = util.batch.universal.system.NodeSetup(memory=JOB_MEMORY_GB, node_kind='f_ocean2', nodes=3, cpus=16, total_cpus_max=6*16, check_for_better=True)
    COST_FUNCTION_NODES_SETUP_JOB = util.batch.universal.system.NodeSetup(memory=40, node_kind='f_ocean2', nodes=1, cpus=1, total_cpus_max=1, walltime=36, check_for_better=True)
    COST_FUNCTION_NODES_SETUP_TRAJECTORY = util.batch.universal.system.NodeSetup(memory=JOB_MEMORY_GB, node_kind='f_ocean2', nodes=1, cpus=12, nodes_max=1, walltime=1, check_for_better=True)

if util.batch.universal.system.IS_NEC:
    COST_FUNCTION_NODES_SETUP_SPINUP = util.batch.universal.system.NodeSetup(memory=JOB_MEMORY_GB, node_kind='clmedium', nodes=8, cpus=16, total_cpus_max=10*24, check_for_better=True)
    COST_FUNCTION_NODES_SETUP_DERIVATIVE = util.batch.universal.system.NodeSetup(memory=JOB_MEMORY_GB, node_kind='clmedium', nodes=3, cpus=16, total_cpus_max=6*24, check_for_better=True)
    COST_FUNCTION_NODES_SETUP_JOB = util.batch.universal.system.NodeSetup(memory=50, node_kind='clmedium', nodes=1, cpus=1, total_cpus_max=1, check_for_better=True, walltime=36)
    COST_FUNCTION_NODES_SETUP_TRAJECTORY = util.batch.universal.system.NodeSetup(memory=JOB_MEMORY_GB, node_kind='clmedium', nodes=1, cpus=1, nodes_max=1, total_cpus_max=16, check_for_better=True, walltime=1)

