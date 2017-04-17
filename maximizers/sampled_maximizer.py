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
        self.population_size = min(population_size, self.get_grid_size(param_distribution))
        self.param_distribution = param_distribution

    def maximize(self, validated_params, validated_scores, current_best_score):

        # Select a sample of parameters
        sampled_params = ParameterSampler(self.param_distribution, self.population_size)

        # Determine the best parameters
        return self.maximize_on_sample(sampled_params, validated_params, validated_scores, current_best_score)

    def maximize_on_sample(self, sampled_params, validated_params, validated_scores, current_best_score):

        if len(validated_scores) == 0:
            return np.random.choice([i for i in sampled_params]), 0

        # Fit parameters
        converted_settings = Converter.convert_settings(validated_params, self.param_distribution)
        try:
            self.gp_score.fit(converted_settings, validated_scores)
        except LinAlgError as e:
            print(str(e))

        best_score = - np.inf
        best_setting = None
        for setting in sampled_params:
            score = self.get_ei(Converter.convert_setting(setting, self.param_distribution), current_best_score)
            if score > best_score:
                best_score = score
                best_setting = setting

        return best_setting, self.realize(best_setting, validated_params, validated_scores, current_best_score, best_score)

    def realize(self, best_setting, validated_params, validated_scores, current_best_score, original):
        params, scores = Converter.drop_zero_scores(validated_params, validated_scores)

        if len(scores) == 0:
            return original

        converted_settings = Converter.convert_settings(params, self.param_distribution)
        self.gp_score.fit(converted_settings, scores)
        realistic_score = self.get_ei(Converter.convert_setting(best_setting, self.param_distribution),
                                      current_best_score)

        return realistic_score

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
