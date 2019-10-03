import os.path

from simulation.constants import SIMULATION_OUTPUT_DIR

# base dir

BASE_DIR = os.path.join(os.path.dirname(SIMULATION_OUTPUT_DIR), 'plots')

PLOT_DIR = os.path.join(BASE_DIR, 'simulation', 'model_{model_name}', 'time_step_{time_step}_-_concentrations_index_{concentrations_index}_-_parameters_index_{parameters_index}', '{kind}')

PLOT_DEFAULT_FILE_EXTENSION = 'svg'
PLOT_FILE_WITHOUT_FILE_EXTENSION = os.path.join(PLOT_DIR, '{plot_name}')
PLOT_FILE = PLOT_FILE_WITHOUT_FILE_EXTENSION + '.{file_extension}'
