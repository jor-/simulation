import util.batch.universal.system

from simulation.model.constants import JOB_MEMORY_GB

COST_FUNCTION_NAMES = ('OLS', 'WLS', 'GLS', 'LOLS', 'LWLS', 'LGLS')

# cache file names

COST_FUNCTION_DIRNAME = 'cost_function'
COST_FUNCTION_F_FILENAME = 'f_-_normalized_{normalized}.txt'
COST_FUNCTION_DF_FILENAME = 'df_-_normalized_{normalized}_-_include_total_concentration_{include_total_concentration}_-_derivative_order_{derivative_order}.txt'

# minimal concentration for log normal distributions

CONCENTRATION_MIN_VALUE = 10**(-6)

# node setups

if util.batch.universal.system.IS_RZ:
    NODES_SETUP_JOB = util.batch.universal.system.NodeSetup(node_kind='f_ocean2', nodes=1, cpus=1, total_cpus_max=1, check_for_better=True)

if util.batch.universal.system.IS_NEC:
    NODES_SETUP_JOB = util.batch.universal.system.NodeSetup(node_kind='clmedium', nodes=1, cpus=1, total_cpus_max=1, check_for_better=True)
