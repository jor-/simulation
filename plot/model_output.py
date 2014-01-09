from util.plot import set_font_size
from interface import plot_model_output

import logging
logging.basicConfig(level=logging.DEBUG)

set_font_size(size=20)
plot_model_output(vmax=(None, 3))