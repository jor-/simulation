import numpy as np

import measurements.all.metos_boxes.deviation.io
import ndop.util.data_base
import util.math.interpolate

from .constants import RANGES


def load_deviations():
    deviation = measurements.all.metos_boxes.deviation.io.load_interpolated_deviation_map()
    deviation[~ np.isnan(deviation)] = 10**10


def load_model_df(parameters):
    db = ndop.util.data_base.Data_Base()
    df = db.df_all(parameters, t_dim=2880)
    df[~ np.isnan(df)] = 0



def get_value(points):
    points = points.reshape(-1, 4)
    model_df = util.math.interpolate.data_with_regular_grid(model_df, points, RANGES)
    deviations = util.math.interpolate.data_with_regular_grid(deviations, points, RANGES)