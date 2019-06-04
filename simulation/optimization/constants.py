import util.batch.universal.system

from simulation.model.constants import JOB_MEMORY_GB

# cache file names

COST_FUNCTION_DIRNAME = 'cost_function'
COST_FUNCTION_F_FILENAME = 'f_-_normalized_{normalized}.txt'
COST_FUNCTION_DF_FILENAME = 'df_-_normalized_{normalized}_-_{derivative_kind}.txt'

# minimal concentration for log normal distributions

CONCENTRATION_MIN_VALUE = 10**(-6)

# node setups

if util.batch.universal.system.IS_RZ:
    COST_FUNCTION_NODES_SETUP_SPINUP = util.batch.universal.system.NodeSetup(memory=JOB_MEMORY_GB, node_kind='f_ocean2', nodes=6, cpus=16, nodes_max=9, check_for_better=True)
    COST_FUNCTION_NODES_SETUP_DERIVATIVE = util.batch.universal.system.NodeSetup(memory=JOB_MEMORY_GB, node_kind='f_ocean2', nodes=3, cpus=16, nodes_max=9, check_for_better=True)
    COST_FUNCTION_NODES_SETUP_TRAJECTORY = util.batch.universal.system.NodeSetup(memory=JOB_MEMORY_GB, node_kind='f_ocean2', nodes=1, cpus=12, nodes_max=1, walltime=2, check_for_better=True)
    COST_FUNCTION_NODES_SETUP_JOB = util.batch.universal.system.NodeSetup(memory=30, node_kind='f_ocean2', nodes=1, cpus=1, total_cpus_max=1, walltime=48, check_for_better=True)

if util.batch.universal.system.IS_NEC:
    COST_FUNCTION_NODES_SETUP_SPINUP = util.batch.universal.system.NodeSetup(memory=JOB_MEMORY_GB, node_kind='clmedium', nodes=4, cpus=32, nodes_max=8, check_for_better=True)
    COST_FUNCTION_NODES_SETUP_DERIVATIVE = util.batch.universal.system.NodeSetup(memory=JOB_MEMORY_GB, node_kind='clmedium', nodes=4, cpus=32, nodes_max=8, check_for_better=True)
    COST_FUNCTION_NODES_SETUP_TRAJECTORY = util.batch.universal.system.NodeSetup(memory=JOB_MEMORY_GB, node_kind='clmedium', nodes=1, cpus=1, nodes_max=1, walltime=1, check_for_better=True)
    COST_FUNCTION_NODES_SETUP_JOB = util.batch.universal.system.NodeSetup(memory=30, node_kind='clmedium', nodes=1, cpus=1, total_cpus_max=1, walltime=48, check_for_better=True)
