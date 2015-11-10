import logging
logging.basicConfig(level=logging.DEBUG)

import ndop.plot.interface
ndop.plot.interface.optimization_parameters_for_data_kind(data_kind='WOD_TMM_1', with_line_search_steps=True)