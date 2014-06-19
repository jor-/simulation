from util.logging import Logger
from interface import model_confidence

with Logger():
    model_confidence(parameter_set_nr=99, vmax=(0.006, 0.006))