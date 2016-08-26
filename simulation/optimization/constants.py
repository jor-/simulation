import numpy as np

import util.batch.universal.system

from simulation.model.constants import JOB_MEMORY_GB

COST_FUNCTION_DIRNAME = 'cost_functions'
COST_FUNCTION_F_FILENAME = 'f.npy'
COST_FUNCTION_F_NORMALIZED_FILENAME = 'f_normalized.npy'
COST_FUNCTION_DF_FILENAME = 'df_-_step_size_{step_size:g}_-_{derivative_kind}.npy'

CONCENTRATION_MIN_VALUE = 10**(-6)


if util.batch.universal.system.IS_RZ:
    COST_FUNCTION_NODES_SETUP_SPINUP = util.batch.universal.system.NodeSetup(memory=JOB_MEMORY_GB, node_kind='f_ocean2', nodes=6, cpus=16, total_cpus_max=9*16, check_for_better=True)
    COST_FUNCTION_NODES_SETUP_DERIVATIVE = util.batch.universal.system.NodeSetup(memory=JOB_MEMORY_GB, node_kind='f_ocean2', nodes=3, cpus=16, total_cpus_max=9*16, check_for_better=True)
    COST_FUNCTION_NODES_SETUP_TRAJECTORY = util.batch.universal.system.NodeSetup(memory=JOB_MEMORY_GB, node_kind='f_ocean2', nodes=1, cpus=12, nodes_max=1, walltime=1, check_for_better=True)
    COST_FUNCTION_NODES_SETUP_JOB = util.batch.universal.system.NodeSetup(memory=30, node_kind='f_ocean2', nodes=1, cpus=1, total_cpus_max=1, walltime=48, check_for_better=True)

if util.batch.universal.system.IS_NEC:
    COST_FUNCTION_NODES_SETUP_SPINUP = util.batch.universal.system.NodeSetup(memory=JOB_MEMORY_GB, node_kind='clfocean', nodes=4, cpus=16, total_cpus_max=8*24, check_for_better=True)
    COST_FUNCTION_NODES_SETUP_DERIVATIVE = util.batch.universal.system.NodeSetup(memory=JOB_MEMORY_GB, node_kind='clmedium', nodes=4, cpus=16, total_cpus_max=8*24, check_for_better=True)
    COST_FUNCTION_NODES_SETUP_TRAJECTORY = util.batch.universal.system.NodeSetup(memory=JOB_MEMORY_GB, node_kind='clexpress', nodes=1, cpus=1, nodes_max=1, total_cpus_max=16, check_for_better=True, walltime=1)
    COST_FUNCTION_NODES_SETUP_JOB = util.batch.universal.system.NodeSetup(memory=30, node_kind='clmedium', nodes=1, cpus=1, total_cpus_max=1, walltime=48, check_for_better=True)

