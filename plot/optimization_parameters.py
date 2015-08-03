import logging
logging.basicConfig(level=logging.DEBUG)

from ndop.plot.interface import optimization_parameters
optimization_parameters(with_line_search_steps=True)