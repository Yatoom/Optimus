from numpy.linalg import LinAlgError
from scipy.stats import norm
from sklearn import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.model_selection import ParameterSampler

from optimus.converter import Converter
import numpy as np


class Maximizer:
    def __init__(self, param_distribution, population_size=100):
        """
        A maximizer for finding the next best point to evaluate
        :param param_distribution: A dictionary of parameters and their distribution of values
        :param population_size: The number of samples to randomly draw from the hyper parameter, to use for finding the
        next best point.
        """

        self.param_distribution = param_distribution
        self.population_size = min(population_size, self.get_grid_size(param_distribution))

        # Setting up the Gaussian Process Regressor
        cov_amplitude = ConstantKernel(1.0, (0.01, 1000.0))
        other_kernel = Matern(
            length_scale=np.ones(len(self.param_distribution)),
            length_scale_bounds=[(0.01, 100)] * len(self.param_distribution),
            nu=2.5)

        gp = GaussianProcessRegressor(
            kernel=cov_amplitude * other_kernel,
            normalize_y=True, random_state=3, alpha=0.0,
            n_restarts_optimizer=2)

        self.gp_score = clone(gp)  # type: GaussianProcessRegressor
        self.gp_time = clone(gp)  # type: GaussianProcessRegressor

        # Initiating information about evaluated parameters
        self.validated_params = []
        self.converted_params = []
        self.validated_scores = []
        self.validated_times = []
        self.current_best_score = -np.inf
        self.current_best_time = np.inf

    def update(self, validated_params, validated_scores, validated_times, current_best_score, current_best_time):
        """
        Updates the maximizer with the latest information about the evaluated parameters.
        :param validated_params: List of validated parameters
        :param validated_scores: List of validation scores
        :param validated_times: List of validation times
        :param current_best_score: The current best score
        :param current_best_time: The current best time in seconds
        """
        self.validated_params = validated_params
        self.converted_params = Converter.convert_settings(validated_params, self.param_distribution)
        self.validated_scores = validated_scores
        self.validated_times = validated_times
        self.current_best_score = current_best_score
        self.current_best_time = current_best_time

    def maximize(self):
        """
        Finds the next best setting to evaluate.
        :return: The best setting and its expected improvement 
        """

        # Select a sample of parameters
        sampled_params = ParameterSampler(self.param_distribution, self.population_size)

        # Determine the best parameters
        return self.maximize_on_sample(sampled_params)

    def maximize_on_sample(self, sampled_params):
        """
        Finds the next best setting to evaluate from a set of samples.
        :param sampled_params: The samples to calculate expected improvement on
        :return: The sample with the highest expected improvement
        """

        if len(self.validated_scores) == 0:
            return np.random.choice([i for i in sampled_params]), 0

        # Fit parameters
        try:
            self.gp_score.fit(self.converted_params, self.validated_scores)
            self.gp_time.fit(self.converted_params, self.validated_times)
        except LinAlgError as e:
            print(str(e))

        best_score = - np.inf
        best_setting = None
        for setting in sampled_params:
            converted_setting = Converter.convert_setting(setting, self.param_distribution)
            score = self.get_ei(converted_setting, self.current_best_score)
            if score > best_score:
                best_score = score
                best_setting = setting

        return best_setting, self.realize(best_setting, best_score)

    def realize(self, best_setting, original):
        """
        Gives a more realistic estimate of the expected improvement by removing validations that resulted in a timeout.
        These are useful to direct the Gaussian Process away from these points, but if we need a realistic estimation 
        of the expected improvement, we should remove these points.
        :param best_setting: The setting to calculate a realistic estimate of
        :param original: The original estimate, will be returned if we can not calculate the realistic estimate
        :return: The realistic estimate
        """
        params, scores = Converter.remove_timeouts(self.validated_params, self.validated_scores)
        times, _ = Converter.remove_timeouts(self.validated_times, self.validated_scores)

        if len(scores) == 0:
            return original

        converted_settings = Converter.convert_settings(params, self.param_distribution)
        self.gp_score.fit(converted_settings, scores)
        self.gp_time.fit(converted_settings, times)

        setting = Converter.convert_setting(best_setting, self.param_distribution)

        return self.get_ei(setting, self.current_best_score)

    def get_ei_per_second(self, point, current_best_score):
        """
        Calculates the expected improvement and divides it by the square root of the estimated validation time.
        :param point: Setting to predict on
        :param current_best_score: The current score optimum
        :return: EI / sqrt(estimated seconds)
        """
        seconds = self.gp_time.predict(point)
        ei = self.get_ei(point, current_best_score)
        return ei / np.sqrt(seconds)

    def get_ei(self, point, current_best_score):
        """
        Calculates the expected improvement.
        :param point: Parameter setting for the GP to predict on
        :param current_best_score: The current score optimum
        :return: Expected improvement value
        """
        point = np.array(point).reshape(1, -1)
        mu, sigma = self.gp_score.predict(point, return_std=True)
        best_score = current_best_score
        mu = mu[0]
        sigma = sigma[0]

        # We want our mu to be higher than the best score
        # We subtract 0.01 because http://haikufactory.com/files/bayopt.pdf
        # (2.3.2 Exploration-exploitation trade-of)
        # Intuition: makes diff less important, while sigma becomes more important
        diff = mu - best_score - 0.01

        # When exploring, we should choose points where the surrogate variance is large.
        if sigma == 0:
            return 0

        # Expected improvement function
        Z = diff / sigma
        ei = diff * norm.cdf(Z) + sigma * norm.pdf(Z)

        return ei

    @staticmethod
    def get_grid_size(param_grid):
        """
        Calculates the grid size (i.e. the number of possible combinations).
        :param param_grid: A dictionary of parameters and their lists of values
        :return: Integer size of the grid
        """
        result = 1
        for i in param_grid.values():
            result *= len(i)
        return result
