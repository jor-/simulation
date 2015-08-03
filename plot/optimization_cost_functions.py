import logging
logging.basicConfig(level=logging.DEBUG)

from ndop.plot.interface import optimization_cost_functions
optimization_cost_functions(y_max=None, with_line_search_steps=True)