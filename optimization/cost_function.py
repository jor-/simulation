import numpy as np

import os
import warnings

import logging
logger = logging.getLogger(__name__)

import measurements.all.woa.data
import measurements.all.pw.data
from ndop.model.eval import Model

import util.io
import util.optimize



class Base:
    
    def __init__(self, years, tolerance=0, combination='and', time_step=1, df_accuracy_order=1, max_nodes_file=None, summand=0, factor=1, job_name_prefix=''):
        from ndop.model.constants import MODEL_PARAMETER_DIM
        
        logger.debug('Initiating {} with {} years, {} tolerance, combination "{}", time step {}, df_accuracy_order {}, summand {}, factor {} and {} as max_nodes_file.'.format(self.__class__.__name__, years, tolerance, combination, time_step, df_accuracy_order, summand, factor, max_nodes_file))
        
        self.years = years
        self.tolerance = tolerance
        self.combination = combination
        self.time_step = time_step
        self.df_accuracy_order = df_accuracy_order
        self.summand = summand
        self.factor = factor
        
        self.model = Model(max_nodes_file=max_nodes_file, job_name_prefix=job_name_prefix)
        
        self.last_parameters_f = None
        self.last_parameters_df = None
        self.last_model_f = None
        self.last_model_df = None
    
    
    ## access to model
    def calculate_model_f(self, parameters):
        raise NotImplementedError("Please implement this method")
    
    def get_model_f(self, parameters):
        if self.last_model_f is not None and all(parameters == self.last_parameters_f):
            logger.debug('Returning cached model f.')
            model_f = self.last_model_f
        else:
            logger.debug('Calculating new model f.')
            model_f = self.calculate_model_f(parameters)
#             model_f += self.summand
#             model_f *= self.factor
            self.last_parameters_f = parameters
            self.last_model_f = model_f
        
        return model_f
    
    
    def calculate_model_df(self, parameters):
        raise NotImplementedError("Please implement this method")
    
    def get_model_df(self, parameters):
        if self.last_model_df is not None and all(parameters == self.last_parameters_df):
            logger.debug('Returning cached model df.')
            model_df = self.last_model_df
        else:
            logger.debug('Calculating new model df.')
            model_df = self.calculate_model_df(parameters)
#             model_df *= self.factor
            self.last_parameters_df = parameters
            self.last_model_df = model_df
        
        return model_df
    
    
    ## access to cache
    def get_file(self, parameters, filename):
        from .constants import COST_FUNCTIONS_DIRNAME
        
        parameter_set_dir = self.model.get_parameter_set_dir(self.time_step, parameters, create=False)
        
        if parameter_set_dir is not None:
            cost_function_dir = os.path.join(parameter_set_dir, COST_FUNCTIONS_DIRNAME, self.__class__.__name__)
            os.makedirs(cost_function_dir, exist_ok=True)
            file = os.path.join(cost_function_dir, filename)
        else:
            file = None
        
        return file
    
    
    def load_file(self, parameters, filename):
        file = self.get_file(parameters, filename)
        if file is not None and os.path.exists(file):
            values = np.load(file)
            logger.debug('Got values from {}.'.format(file))
        else:
            values = None
        return values
    
    def save_file(self, parameters, filename, values):
        file = self.get_file(parameters, filename)
        util.io.save_npy_and_txt(values, file)
        logger.debug('Saved values to {}.'.format(file))
    
    
    def matches_options(self, parameters, options_file):
        options = self.load_file(parameters, options_file)
        if options is not None:
            if options[2]:
                matches = options[0] >= self.years and options[1] <= self.tolerance
            else:
                matches = options[0] >= self.years or options[1] <= self.tolerance
            
            if len(options) == 4:
                matches = matches and options[3] >= self.df_accuracy_order
        else:
            matches = False
        
        return matches
        
    
    
    def calculate_f(self, parameters):
        raise NotImplementedError("Please implement this method")
    
    def get_f(self, parameters):
        from .constants import COST_FUNCTION_F_FILENAME, COST_FUNCTION_F_OPTION_FILENAME
        
#         ## check if cached value is matching
#         options = self.load_file(parameters, COST_FUNCTION_F_OPTION_FILENAME)
# #         matches = options is not None and options[0] >= self.years and options[1] <= self.tolerance and (options[2] or self.combination == 'or')
#         if options is not None:
#             if options[2]:
#                 matches = options[0] >= self.years and options[1] <= self.tolerance
#             else:
#                 matches = options[0] >= self.years or options[1] <= self.tolerance
#         else:
#             matches = False
            
        
        ## if matching load value
        if self.matches_options(parameters, COST_FUNCTION_F_OPTION_FILENAME):
            f = self.load_file(parameters, COST_FUNCTION_F_FILENAME)
            logger.debug('Cached f value {} loaded.'.format(f))
        ## else calculate and save value
        else:
            f = self.calculate_f(parameters)
            f += self.summand
            f *= self.factor
            logger.debug('f value {} calculated and saving.'.format(f))
            self.save_file(parameters, COST_FUNCTION_F_FILENAME, f)
            options = (self.years, self.tolerance, self.combination == 'and')
            self.save_file(parameters, COST_FUNCTION_F_OPTION_FILENAME, options)
        
        return f
    
    
    def calculate_df(self, parameters):
        raise NotImplementedError("Please implement this method")
    
    def get_df(self, parameters):
        from .constants import COST_FUNCTION_DF_FILENAME, COST_FUNCTION_DF_OPTION_FILENAME
        
#         ## check if cached value is matching
#         options = self.load_file(parameters, COST_FUNCTION_DF_OPTION_FILENAME)
#         matches = options is not None and options[0] >= self.years and options[1] <= self.tolerance and (options[2] or self.combination == 'or') and options[3] >= self.df_accuracy_order
#         
#         if options is not None:
#             if options[2]:
#                 matches = options[0] >= self.years and options[1] <= self.tolerance
#             else:
#                 matches = options[0] >= self.years or options[1] <= self.tolerance
#         else:
#             matches = False
#         matches = matches and options[3] >= self.df_accuracy_order
        
        ## if matching load value
        if self.matches_options(parameters, COST_FUNCTION_DF_OPTION_FILENAME):
            df = self.load_file(parameters, COST_FUNCTION_DF_FILENAME)
            logger.debug('Cached df value loaded.')
        ## else calculate and save value
        else:
            df = self.calculate_df(parameters)
            df *= self.factor
            logger.debug('df value calculated and saving.')
            self.save_file(parameters, COST_FUNCTION_DF_FILENAME, df)
            options = (self.years, self.tolerance, self.combination == 'and', self.df_accuracy_order)
            self.save_file(parameters, COST_FUNCTION_DF_OPTION_FILENAME, options)
        
        return df
    



class WOA_Base(Base):
    
    def __init__(self, years, tolerance=0, combination='and', time_step=1, df_accuracy_order=1, max_nodes_file=None, summand=0, job_name_prefix=''):
        self.means = measurements.all.woa.data.means()
        nobs = measurements.all.woa.data.nobs()
        factor = 1 / (np.nansum(nobs))
        
        super().__init__(years, tolerance, combination, time_step, df_accuracy_order=df_accuracy_order, max_nodes_file=max_nodes_file, summand=summand, factor=factor, job_name_prefix=job_name_prefix)
        
    
    def calculate_model_f(self, parameters):
        model_f = self.model.get_f_for_all(parameters, time_dim_desired=12, years=self.years, tolerance=self.tolerance, combination=self.combination, time_step=self.time_step)
        model_f = np.asanyarray(model_f)
        return model_f
    
    
    def calculate_model_df(self, parameters):
        model_df = self.model.get_df_for_all(parameters, time_dim_desired=12, years=self.years, tolerance=self.tolerance, combination=self.combination, time_step=self.time_step, accuracy_order=self.df_accuracy_order)
        model_df = np.asanyarray(model_df)
        model_df = np.swapaxes(model_df, 0, 1)
        return model_df



class WOA_LS(WOA_Base):
    
    def __init__(self, years, tolerance=0, combination='and', time_step=1, df_accuracy_order=1, max_nodes_file=None):
        super().__init__(years, tolerance, combination, time_step, df_accuracy_order=df_accuracy_order, max_nodes_file=max_nodes_file, job_name_prefix='WOA_OLS_')
        
        nobs = measurements.all.woa.data.nobs()
        varis = measurements.all.woa.data.varis()
        inverse_variances = nobs / varis
        
        mask = inverse_variances > 0
        average_inverse_variance = inverse_variances[mask].mean()
        
        n = mask.sum()
        ln_det_covariance_matrix = - n * np.log(average_inverse_variance)
        
        self.average_inverse_variance = average_inverse_variance
    
    
    def calculate_f(self, parameters):
        model_f = self.get_model_f(parameters)
        
        means = self.means
        average_inverse_variance = self.average_inverse_variance
        
        f = average_inverse_variance * np.nansum((means - model_f)**2)
        
        return f
    
    
    def calculate_df(self, parameters):
        model_f = self.get_model_f(parameters)
        model_df = self.get_model_df(parameters)
        
        means = self.means
        average_inverse_variance = self.average_inverse_variance
        
        df_factors = means - model_f
        
        p_dim = len(parameters)
        df = np.empty(p_dim)
        
        for i in range(p_dim):
            df[i] = np.nansum(df_factors * model_df[i])
        
        df *= - 2 * average_inverse_variance
        
        return df



class WOA_WLS(WOA_Base):
    
    def __init__(self, years, tolerance=0, combination='and', time_step=1, df_accuracy_order=1, max_nodes_file=None):
        ## load data
#         self.means = measurements.all.woa.data.means()
#         nobs = measurements.all.woa.data.nobs()
        nobs = measurements.all.woa.data.nobs()
        varis = measurements.all.woa.data.varis()
        inverse_variances = nobs / varis
        self.inverse_variances = inverse_variances
        
        
        ## calculate summand
        mask = inverse_variances > 0
        ln_det_covariance_matrix = - np.sum(np.log(inverse_variances[mask]))
        
        ## super init
        super().__init__(years, tolerance, combination, time_step, df_accuracy_order=df_accuracy_order, max_nodes_file=max_nodes_file, summand=ln_det_covariance_matrix, job_name_prefix='WOA_WLS_')
#         self.factor = 1 / ((nobs > 0).sum() - MODEL_PARAMETER_DIM)
#         self.factor = 1 / (np.nansum(nobs) - MODEL_PARAMETER_DIM)
    
    
    def calculate_f(self, parameters):
        model_f = self.get_model_f(parameters)
        
        means = self.means
        inverse_variances = self.inverse_variances
        
        f = np.nansum(inverse_variances * (means - model_f)**2)
        
        return f
    
    
    def calculate_df(self, parameters):
        model_f = self.get_model_f(parameters)
        model_df = self.get_model_df(parameters)
        
        means = self.means
        inverse_variances = self.inverse_variances
        
        df_factors = inverse_variances * (means - model_f)
        
        p_dim = len(parameters)
        df = np.empty(p_dim)
        
        for i in range(p_dim):
            df[i] = np.nansum(df_factors * model_df[i])
        
        df *= - 2 
        
        return df

    




class WOD_Base(Base):
    
    def __init__(self, years, tolerance=0, combination='and', time_step=1, df_accuracy_order=1, max_nodes_file=None, summand=0, job_name_prefix=''):
        points, values = measurements.all.pw.data.get_points_and_values()
        
        self.points = points
        self.values = values
        
        nobs = sum(map(len, points))
        factor = 1 / nobs
        
        super().__init__(years, tolerance, combination, time_step, df_accuracy_order=df_accuracy_order, max_nodes_file=max_nodes_file, summand=summand, factor=factor, job_name_prefix=job_name_prefix)
    
    
    def calculate_model_f(self, parameters):
        model_f = self.model.get_f_for_points(parameters, self.points, years=self.years, tolerance=self.tolerance, combination=self.combination, time_step=self.time_step)
        return model_f
    
    
    def calculate_model_df(self, parameters):
        model_df = self.model.get_df_for_points(parameters, self.points, years=self.years, tolerance=self.tolerance, combination=self.combination, time_step=self.time_step, accuracy_order=self.df_accuracy_order)
        return model_df



class WOD_LS(WOD_Base):
    
    def __init__(self, years, tolerance=0, combination='and', time_step=1, df_accuracy_order=1, max_nodes_file=None):
        super().__init__(years, tolerance, combination, time_step, df_accuracy_order=df_accuracy_order, max_nodes_file=max_nodes_file, job_name_prefix='WOD_OLS_')
    
    
    def calculate_f(self, parameters):
        model_f = self.get_model_f(parameters)
        
        values = self.values
        f = 0
        
        for tracer_index in range(len(values)):
            f += np.sum((values[tracer_index] - model_f[tracer_index])**2)
        
        return f
    
    
    def calculate_df(self, parameters):
        model_f = self.get_model_f(parameters)
        model_df = self.get_model_df(parameters)
        
        values = self.values
        
        p_dim = len(parameters)
        df = np.zeros(p_dim)
        
        for tracer_index in range(len(values)):
            df_factors = values[tracer_index] - model_f[tracer_index]
            for parameter_index in range(p_dim):
                df[parameter_index] += np.sum(df_factors * model_df[tracer_index][parameter_index])
        
        df *= - 2
        
        return df



class WOD_WLS(WOD_Base):
    
    def __init__(self, years, tolerance=0, combination='and', time_step=1, df_accuracy_order=1, max_nodes_file=None):
        ## calculate variance
        (dop_deviation, po4_deviation) = measurements.all.pw.data.get_deviation()
        self.variance = (dop_deviation**2, po4_deviation**2)
        
        ## calculate summand
        summand = 0
        for deviation in (dop_deviation, po4_deviation):
            summand += np.sum(np.log(deviation))
        summand *= 2
        
        ## super init
        super().__init__(years, tolerance, combination, time_step, df_accuracy_order=df_accuracy_order, max_nodes_file=max_nodes_file, summand=summand, job_name_prefix='WOD_WLS_')
    
    
    def calculate_f(self, parameters):
        model_f = self.get_model_f(parameters)
        
        values = self.values
        factor = self.factor
        variance = self.variance
        f = 0
        
        for tracer_index in range(len(values)):
            f += np.sum((values[tracer_index] - model_f[tracer_index])**2 / variance[tracer_index])
        
        f *= factor
        
        return f
    
    
    def calculate_df(self, parameters):
        model_f = self.get_model_f(parameters)
        model_df = self.get_model_df(parameters)
        
        values = self.values
        factor = self.factor
        variance = self.variance
        
        p_dim = len(parameters)
        df = np.zeros(p_dim)
        
        for tracer_index in range(len(values)):
            df_factors = (values[tracer_index] - model_f[tracer_index]) / variance[tracer_index]
            for parameter_index in range(p_dim):
                df[parameter_index] += np.sum(df_factors * model_df[tracer_index][parameter_index])
        
        df *= - 2 * factor
        
        return df



class WOD_GLS(WOD_Base):
    
    def __init__(self, years, tolerance=0, combination='and', time_step=1, df_accuracy_order=1, max_nodes_file=None):
        ## calculate summand
        (dop_deviation, po4_deviation) = measurements.all.pw.data.get_deviation()
        self.deviation = (dop_deviation, po4_deviation)
        
        summand = 0
        for deviation in self.deviation:
            summand += np.sum(np.log(deviation))
        summand *= 2
#         self.summand = summand
        
        ## super init
        super().__init__(years, tolerance, combination, time_step, df_accuracy_order=df_accuracy_order, max_nodes_file=max_nodes_file, summand=summand, job_name_prefix='WOD_GLS_')
        
        ## setup correlation bounds and last correlations
        dim = 3
#         lower_bound = 10**(-5)
        upper_bound = 1 - 10**(-5)
        lower_bound = - upper_bound
        bounds = ((lower_bound, upper_bound),) * dim
        self.correlation_parameters_bounds = bounds
        
        last = np.array((0.1,)*dim)
        self.last_correlation_parameters = last
    
    
    
    def f_for_cp_and_diff(self, correlation_parameters, diff_sum, diff_sum_quad):
        ## check input
        if not np.all(np.logical_and(correlation_parameters > -1, correlation_parameters < 1)):
            raise ValueError('Each correlation parameter have to be in (-1, 1), but they are {}.'.format(correlation_parameters))
        
        ## calculate determinante of correlation matrix
        n = len(self.values[0])
        m = len(self.values[1])
        a = correlation_parameters[0]
        b = correlation_parameters[1]
        c = correlation_parameters[2]
        
        ln_det = (n-1) * np.log(1-a) + (m-1) * np.log(1-b) + np.log(1 + (n-1)*a) + np.log(1 - b + m * (b - n * c**2 * (1-(n-1)*a)))
        if ln_det is np.isnan(ln_det):
            warnings.warn('Correlation matrix is singular for m={}, n={}, a={}, b={} and c={}.'.format(m, n, a, b, c))
            ln_det = np.inf
            
        
        ## calculate diff.T * covariance_matrix.I * diff
        diff_sum = np.matrix(diff_sum).T
        diff_sum_quad = np.matrix(diff_sum_quad).T
        
        A = np.matrix([[a, c], [c, b]])
        W = np.matrix([[1-a, 0], [0, 1-b]])
        D = np.matrix([[n, 0], [0, m]])
#         H = W * A.I * W + D * W
#         
#         product_value = diff_sum_quad.T * W.I * diff_sum_quad - diff_sum.T * H.I * diff_sum
        H = W + D * A
        
        product_value = diff_sum_quad.T * W.I * diff_sum_quad - diff_sum.T * W.I * A * H.I * diff_sum
        product_value = product_value.item()
          
        ## calulate function value
        f = ln_det + product_value
        
        logger.debug('Returning ln_det {} and product_value {}.'.format(ln_det, product_value))
        
        return f
    
    
    
    def f_with_opt_cp_for_diff(self, diff_sum, diff_sum_quad):
        f = lambda cp: self.f_for_cp_and_diff(cp, diff_sum, diff_sum_quad)
        last_correlation_parameters = self.last_correlation_parameters
        x0s = (last_correlation_parameters, (0, 0, 0))
        bounds = self.correlation_parameters_bounds
        
        (opt_correlation_parameters, opt_f) = util.optimize.minimize(f, x0s, bounds=bounds, global_method='basin_hopping', global_iterations=15)
        
        self.last_correlation_parameters = opt_correlation_parameters
        
        logger.debug('Returning optimal correlation parameters {} with value {}.'.format(opt_correlation_parameters, opt_f))
        
        return opt_f
    
    
    
    
    def f_with_opt_cp_for_model_parameters(self, model_parameters):
        from .constants import COST_FUNCTION_CORRELATION_PARAMETER_FILENAME
        
        ## calculate diff norms
        model_f = self.get_model_f(model_parameters)
        values = self.values
        deviation = self.deviation
        
        tracer_dim = 2
        diff_sum = np.empty(tracer_dim)
        diff_sum_quad = np.empty(tracer_dim)
        
        for i in range(tracer_dim):
            diff = (values[i] - model_f[i]) / deviation[i]
            diff_sum[i] = np.sum(diff)
            diff_sum_quad[i] = np.sum(diff**2)
        
        ## calulate function value
        f = self.f_with_opt_cp_for_diff(diff_sum, diff_sum_quad)
        
        self.save_file(model_parameters, COST_FUNCTION_CORRELATION_PARAMETER_FILENAME, self.last_correlation_parameters)
        
        return f
    
    
    
    def df_with_opt_cp_for_model_parameters(self, model_parameters):
        ## calculate diff norms and its derivatives
        model_f = self.get_model_f(model_parameters)
        model_df = self.get_model_df(model_parameters)
        values = self.values
        deviation = self.deviation
        
        tracer_dim = len(values)
        diff_sum = np.empty(tracer_dim)
        diff_sum_quad = np.empty(tracer_dim)
        p_dim = len(model_parameters)
        d_diff_sum = np.empty([tracer_dim, p_dim])
        d_diff_sum_quad = np.empty([tracer_dim, p_dim])
        
        for i in range(tracer_dim):
            diff = (values[i] - model_f[i]) / deviation[i]
            diff_sum[i] = np.sum(diff)
            diff_sum_quad[i] = np.sum(diff**2)
            for j in range(p_dim):
                d_diff_sum[i, j] = - np.sum(model_df[i][j] / deviation[i])
                d_diff_sum_quad[i, j] = - 2 * np.sum((values[i] - model_f[i]) * model_df[i][j] / deviation[i]**2)
        
        ## calculate function values and its derivatives
        f_p = self.f_with_opt_cp_for_diff(diff_sum, diff_sum_quad)
        d1_f_p = util.optimize.finite_differences(lambda diff_sum:self.f_with_opt_cp_for_diff(diff_sum, diff_sum_quad), diff_sum, f_x=f_p, bounds=None, accuracy_order=1)
        d2_f_p = util.optimize.finite_differences(lambda diff_sum_quad:self.f_with_opt_cp_for_diff(diff_sum, diff_sum_quad), diff_sum_quad, f_x=f_p, bounds=((0, np.inf),)*2, accuracy_order=1)
        
        ## compose derivative
        d_f_p = np.matrix(d1_f_p) * np.matrix(d_diff_sum) + np.matrix(d2_f_p) * np.matrix(d_diff_sum_quad)
        d_f_p = np.array(d_f_p.flat)
        
        return d_f_p;
    
    
    def calculate_f(self, parameters):
        f = self.f_with_opt_cp_for_model_parameters(parameters)
        return f
    
    
    def calculate_df(self, parameters):
        df = self.df_with_opt_cp_for_model_parameters(parameters)
        return df







class Family:
    
    def __init__(self, main_member_class, member_classes, years, tolerance=0, combination='and', time_step=1, df_accuracy_order=1, max_nodes_file=None):
        
        logger.debug('Initiating cost function family with main member {} and members {}.'.format(main_member_class, member_classes))
        
        if main_member_class not in member_classes:
            raise ValueError('The main member class has to be in {}, but its {}.'.format(member_classes, main_member_class))
        
        family = []
        for member_class in member_classes:
            if member_class is not main_member_class:
                family.append(member_class(years, tolerance, combination, time_step, df_accuracy_order=df_accuracy_order, max_nodes_file=max_nodes_file))
        
        self.family = family
        self.main_member = main_member_class(years, tolerance, combination, time_step, df_accuracy_order=df_accuracy_order, max_nodes_file=max_nodes_file)
    
    
    def get_f(self, parameters):
        main_member = self.main_member
        main_member_f = main_member.get_f(parameters)
        last_parameters_f = main_member.last_parameters_f
        last_model_f = main_member.last_model_f
        
        for member in self.family:
            if last_model_f is not None:
                member.last_parameters_f = last_parameters_f
                member.last_model_f = last_model_f
                
            member.get_f(parameters)
            
            if last_model_f is None:
                last_parameters_f = member.last_parameters_f
                last_model_f = member.last_model_f
        
        return main_member_f
    
    
    def get_df(self, parameters):
        main_member = self.main_member
        main_member_df = main_member.get_df(parameters)
        last_parameters_df = main_member.last_parameters_df
        last_model_df = main_member.last_model_df
        
        for member in self.family:
            if last_model_df is not None:
                member.last_parameters_df = last_parameters_df
                member.last_model_df = last_model_df
            
            member.get_df(parameters)
            
            if last_model_df is None:
                last_parameters_df = member.last_parameters_df
                last_model_df = member.last_model_df
        
        return main_member_df



class WOA_Family(Family):
    
    def __init__(self, main_member_class, years, tolerance=0, combination='and', time_step=1, df_accuracy_order=1, max_nodes_file=None):
        member_classes = (WOA_LS, WOA_WLS)
        super().__init__(main_member_class, member_classes, years=years, tolerance=tolerance, combination=combination, time_step=time_step, df_accuracy_order=df_accuracy_order, max_nodes_file=max_nodes_file)



class WOD_Family(Family):
    
    def __init__(self, main_member_class, years, tolerance=0, combination='and', time_step=1, df_accuracy_order=1, max_nodes_file=None):
        member_classes = (WOD_LS, WOD_WLS, WOD_GLS)
#         member_classes = (WOD_LS, WOD_WLS)
#         member_classes = (WOD_LS, )
        super().__init__(main_member_class, member_classes, years=years, tolerance=tolerance, combination=combination, time_step=time_step, df_accuracy_order=df_accuracy_order, max_nodes_file=max_nodes_file)