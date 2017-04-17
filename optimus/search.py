import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.model_selection import cross_val_score
import numpy as np
import warnings

from extra.timeout import Timeout
from maximizers.bounded_maximizer import BoundedMaximizer
from maximizers.sampled_maximizer import SampledMaximizer
from optimus.builder import Builder
from optimus.converter import Converter

warnings.filterwarnings("ignore")


class Search:
    def __init__(self, estimator, param_distributions, n_iter=10, population_size=100, scoring=None, cv=10,
                 n_jobs=-1, verbose=True, maximizer=SampledMaximizer, timeout_score=0):
        # Accept parameters
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.population_size = population_size
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.cv = cv
        self.verbose = verbose
        self.param_keys = param_distributions.keys()
        self.timeout_score = timeout_score

        # Initialization of validated values, scores and the current best setting
        self.validated_params = []
        self.validated_scores = []
        self.current_best_setting = None
        self.current_best_score = 0

        cov_amplitude = ConstantKernel(1.0, (0.01, 1000.0))
        other_kernel = Matern(
            length_scale=np.ones(len(self.param_distributions)),
            length_scale_bounds=[(0.01, 100)] * len(self.param_distributions),
            nu=2.5)

        self.gp = GaussianProcessRegressor(
            kernel=cov_amplitude * other_kernel,
            normalize_y=True, random_state=3, alpha=0.0,
            n_restarts_optimizer=2)

        # Setup maximizer
        self.Maximizer = maximizer(self.gp, self.param_distributions)

    def _say(self, *args):
        if self.verbose:
            print(*args)

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
            best_params, best_score = self.Maximizer.maximize(
                self.validated_params, self.validated_scores, self.current_best_score)
            self.evaluate(best_params, X, y, current_best_score=self.current_best_score)

        return self.get_best()

    def maximize(self, current_best_score):
        return self.Maximizer.maximize(self.validated_params, self.validated_scores, current_best_score)

    def evaluate(self, parameters, X, y, max_eval_time=None, current_best_score=None):

        self._say(
            "Evaluating parameters: %s with timeout %s" % (Converter.readable_parameters(parameters), max_eval_time))

        # Initiate success variable
        success = True

        # Build the estimator
        best_estimator = Builder.build_pipeline(self.estimator, parameters)

        # Try evaluating within a time limit
        with Timeout(max_eval_time):
            try:
                score = cross_val_score(best_estimator, X, y, scoring=self.scoring, cv=self.cv, n_jobs=-1)
            except (GeneratorExit, TimeoutError):  # Note:
                self._say("Timeout :(")
                success = False
                score = [self.timeout_score]

        # Get the mean and store the results
        score = np.mean(score)
        self.validated_scores.append(score)
        self.validated_params.append(parameters)
        self.current_best_score = max(score, self.current_best_score)

        if current_best_score is not None:
            self._say("Score: %s | best = %s" % (score, max(current_best_score, score)))
        else:
            self._say("Score: %s" % score)

        return success
