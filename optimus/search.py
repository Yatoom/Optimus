from copy import copy
from sklearn import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import ParameterSampler, cross_val_score
import numpy as np
from scipy.stats import norm
import warnings

from extra.timeout import Timeout
from optimus.builder import Builder
from optimus.converter import Converter

warnings.filterwarnings("ignore")


class Search:
    def __init__(self, estimator, param_distributions, n_iter=10, population_size=100, scoring=None, cv=10,
                 n_jobs=-1, verbose=True):
        # Accept parameters
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.population_size = min(population_size, self._get_grid_size(param_distributions))
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.cv = cv
        self.verbose = verbose
        self.param_keys = param_distributions.keys()

        # Initialization of validated values, scores and the current best setting
        self.validated_params = []
        self.validated_scores = []
        self.current_best_setting = None
        self.current_best_score = 0

        # Setup Gaussian Processes
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            n_restarts_optimizer=10,
        )

    def _say(self, *args):
        if self.verbose:
            print(*args)

    def sample_and_maximize(self, current_best_score, realistic=False):

        # Select a sample of parameters
        sampled_params = ParameterSampler(self.param_distributions, self.population_size)

        # Determine the best parameters
        best_parameters, best_score = self._maximize(sampled_params, current_best_score, realistic)

        return best_parameters, best_score

    def get_best(self):
        best = np.argmax(self.validated_scores)
        best_setting = self.validated_params[best]

        # Convert best setting values to a key-value dictionary
        self._say("\n===\nBest parameters:", best_setting)

        # Return the estimator with the best parameters
        return Builder.build_pipeline(self.estimator, best_setting)

    def fit(self, X, y=None, groups=None):
        self._say("Bayesian search with %s iterations..." % self.n_iter)
        for i in range(0, self.n_iter):
            self._say("---\nIteration %s/%s" % (i + 1, self.n_iter))
            best_params, best_score = self.sample_and_maximize(self.current_best_score)
            self.evaluate(best_params, X, y)

        return self.get_best()

    def _get_grid_size(self, param_grid):
        result = 1
        for i in param_grid.values():
            result *= len(i)
        return result

    def _maximize(self, sampled_params, current_best_score, realistic=False):
        # Fit parameters
        if len(self.validated_scores) > 0:
            self.gp.fit(Converter.convert_settings(self.validated_params, self.param_distributions), self.validated_scores)
        else:
            return np.random.choice([i for i in sampled_params]), 0

        best_score = -1
        best_setting = None
        for setting in sampled_params:
            score = self._get_ei(Converter.convert_setting(setting, self.param_distributions), current_best_score)
            if score > best_score:
                best_score = score
                best_setting = setting

        if realistic and len(self.validated_scores) > 0:
            params, scores = Converter.drop_zero_scores(self.validated_params, self.validated_scores)
            if len(scores) > 0:
                self.gp.fit(Converter.convert_settings(params, self.param_distributions), scores)
                best_score = self._get_ei(Converter.convert_setting(best_setting, self.param_distributions), current_best_score)

        return best_setting, best_score

    def _get_ei(self, point, current_best_score):
        point = np.array(point).reshape(1, -1)
        mu, sigma = self.gp.predict(point, return_std=True)
        best_score = current_best_score
        mu = mu[0]
        sigma = sigma[0]

        # We want our mu to be lower than the loss optimum
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

    def evaluate(self, parameters, X, y, max_eval_time=None):

        self._say("Evaluating parameters: %s with timeout %s" % (Converter.readable_parameters(parameters), max_eval_time))

        # Initiate success variable
        success = True

        # Build the estimator
        best_estimator = Builder.build_pipeline(self.estimator, parameters)

        # Try evaluating within a time limit
        with Timeout(max_eval_time):
            try:
                score = cross_val_score(best_estimator, X, y, scoring=self.scoring, cv=self.cv, n_jobs=-1)
            except TimeoutError:
                self._say("Timeout :(")
                success = False
                score = [0]

        # Get the mean and store the results
        score = np.mean(score)
        self._say("Score: %s" % score)
        self.validated_scores.append(score)
        self.validated_params.append(parameters)
        self.current_best_score = max(score, self.current_best_score)

        return success
