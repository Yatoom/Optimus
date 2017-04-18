from numpy.linalg import LinAlgError
from scipy.stats import norm
from sklearn import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import ParameterSampler

from optimus.converter import Converter
import numpy as np


class SampledMaximizer:
    def __init__(self, gp, param_distribution, population_size=100):
        self.gp_score = gp  # type: GaussianProcessRegressor
        self.gp_time = clone(gp)  # type: GaussianProcessRegressor
        self.population_size = min(population_size, self.get_grid_size(param_distribution))
        self.param_distribution = param_distribution

        self.validated_params = []
        self.converted_params = []
        self.validated_scores = []
        self.validated_times = []
        self.current_best_score = -np.inf
        self.current_best_time = np.inf

    def tell(self, validated_params, validated_scores, validated_times, current_best_score, current_best_time):
        self.validated_params = validated_params
        self.converted_params = Converter.convert_settings(validated_params, self.param_distribution)
        self.validated_scores = validated_scores
        self.validated_times = validated_times
        self.current_best_score = current_best_score
        self.current_best_time = current_best_time

    def maximize(self):

        # Select a sample of parameters
        sampled_params = ParameterSampler(self.param_distribution, self.population_size)

        # Determine the best parameters
        return self.maximize_on_sample(sampled_params)

    def maximize_on_sample(self, sampled_params):

        if len(self.validated_scores) == 0:
            return np.random.choice([i for i in sampled_params]), 0

        # Fit parameters
        try:
            self.gp_score.fit(self.converted_params, self.validated_scores)
            self.gp_time.fit(self.converted_params, self.validated_times)
        except LinAlgError as e:
            print(str(e))

        best_score = - 100000
        best_setting = None
        for setting in sampled_params:
            converted_setting = Converter.convert_setting(setting, self.param_distribution)
            score = self.get_ei_per_second(converted_setting, self.current_best_score)
            if score > best_score:
                best_score = score
                best_setting = setting

        return best_setting, self.realize(best_setting, best_score)

    def realize(self, best_setting, original):
        params, scores = Converter.drop_zero_scores(self.validated_params, self.validated_scores)
        times, _ = Converter.drop_zero_scores(self.validated_times, self.validated_scores)

        if len(scores) == 0:
            return original

        converted_settings = Converter.convert_settings(params, self.param_distribution)
        self.gp_score.fit(converted_settings, scores)
        self.gp_time.fit(converted_settings, times)

        setting = Converter.convert_setting(best_setting, self.param_distribution)

        return self.get_ei_per_second(setting, self.current_best_score)

    def get_ei_per_second(self, point, current_best_score):
        seconds = self.gp_time.predict(point)
        ei = self.get_ei(point, current_best_score)
        return ei / seconds

    def get_ei(self, point, current_best_score):
        point = np.array(point).reshape(1, -1)
        mu, sigma = self.gp_score.predict(point, return_std=True)
        best_score = current_best_score
        mu = mu[0]
        sigma = sigma[0]

        # We subtract 0.01 because http://haikufactory.com/files/bayopt.pdf
        # (2.3.2 Exploration-exploitation trade-of)
        # Intuition: makes diff less important, while sigma becomes more important

        # We want our mu to be higher than the best score
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
        result = 1
        for i in param_grid.values():
            result *= len(i)
        return result
