import numpy as np

import simulation.accuracy.asymptotic
import simulation.util.data_base

import util.parallel.universal
import util.math.interpolate
import util.math.optimize.with_deap
import util.logging
logger = util.logging.logger


class CostFunction():

    def __init__(self, parameters, time_dim_df=2880, value_mask=None, parallel_mode=util.parallel.universal.MODES['multiprocessing']):
        from .constants import BOUNDS
        self.BOUNDS = np.array(BOUNDS)

        self.parameters =  parameters
        self.time_dim_df = time_dim_df
        self.value_mask = value_mask
        self.parallel_mode = parallel_mode
        self.as_shared_array = parallel_mode == util.parallel.universal.MODES['multiprocessing']

        accuracy = simulation.accuracy.asymptotic.WLS('WOD')
        self.accuracy= accuracy

        db = accuracy.data_base
        self.df = db.df_boxes(parameters, time_dim=time_dim_df, as_shared_array=self.as_shared_array)
        self.inverse_deviations = db.inverse_deviations_boxes(time_dim=52, as_shared_array=self.as_shared_array)


    def f(self, points):
        points = np.asanyarray(points).reshape(-1, 5)

        logger.debug('Calculating cost function for points {}'.format(points))

        points_df = util.math.interpolate.data_with_regular_grid(self.df, points, self.BOUNDS)
        points_df[np.isnan(points_df)] = 0
        points_inverse_deviations = util.math.interpolate.data_with_regular_grid(self.inverse_deviations, points, self.BOUNDS)
        points_inverse_deviations[np.isnan(points_inverse_deviations)] = 10**10

        additional = {'DF': points_df, 'inverse_deviations': points_inverse_deviations}
        information_matrix = self.accuracy.information_matrix(self.parameters, additional)
        average_model_confidence = self.accuracy.average_model_confidence(self.parameters, information_matrix, time_dim_df=self.time_dim_df, value_mask=self.value_mask, parallel_mode=self.parallel_mode)

        logger.debug('Value {} for cost function calculated'.format(average_model_confidence))

        return average_model_confidence


    def optimize(self, number_of_points, number_of_initial_individuals=100, number_of_generations=50):
        logger.debug('Optimizing cost function for {} points'.format(number_of_points))

        bounds = np.tile(self.BOUNDS.T, number_of_points).T
        points_opt = util.math.optimize.with_deap.minimize(self.f, bounds, number_of_initial_individuals=number_of_initial_individuals, number_of_generations=number_of_generations)

        logger.debug('Optimal points {} calculated'.format(points_opt))
        return points_opt
