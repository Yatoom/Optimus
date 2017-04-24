import threading
import time
import traceback

from sklearn.model_selection import cross_val_score
import numpy as np
import warnings

from extra.timeout import Timeout
from optimus.builder import Builder
from optimus.converter import Converter
from optimus.maximizer import Maximizer

warnings.filterwarnings("ignore")


class ModelOptimizer:
    def __init__(self, estimator, param_distributions, n_iter=10, population_size=100, scoring=None, cv=10,
                 verbose=True, timeout_score=0, use_ei_per_second=False):
        """
        An optimizer using Gaussian Processes for optimizing a single model. 
        :param estimator: The estimator to optimize
        :param param_distributions: A dictionary of parameter distributions for the estimator. An extra key 
        `@preprocessor` can be added to try out different preprocessors.
        :param n_iter: The number of iterations of finding and evaluating settings
        :param population_size: The number of samples to randomly draw from the hyper parameter, to use for finding the
        next best point.
        :param scoring: A Sklearn-compatible scoring method
        :param cv: A sklearn-compatible cross-validation object
        :param verbose: The verbosity level (boolean)
        :param timeout_score: The score value to insert in case of timeout 
        :param use_ei_per_second: Whether to use the standard EI or the EI / sqrt(second)
        """

        # Accept parameters
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.population_size = population_size
        self.scoring = scoring
        self.cv = cv
        self.verbose = verbose
        self.param_keys = param_distributions.keys()
        self.timeout_score = timeout_score

        # Initialization of validated values, scores and the current best setting
        self.validated_params = []
        self.validated_scores = []
        self.validated_times = []
        self.current_best_setting = None
        self.current_best_score = -np.inf
        self.current_best_time = np.inf

        # Setup maximizer
        self.Maximizer = Maximizer(param_distribution=self.param_distributions, timeout_score=self.timeout_score,
                                   use_ei_per_second=use_ei_per_second)

    def get_best(self):
        """
        Builds an estimator with the current best parameters.
        :return: A pipeline or a single estimator, set up with the current best parameters.
        """
        best = np.argmax(self.validated_scores)
        best_setting = self.validated_params[best]

        # Convert best setting values to a key-value dictionary
        self._say("\n===\nBest parameters:", best_setting)

        # Return the estimator with the best parameters
        return Builder.build_pipeline(self.estimator, best_setting)

    def fit(self, X, y=None, max_eval_time=200):
        """
        Fit method for stand-alone use of the Optimus single-model optimizer.
        :param max_eval_time: Maximum evaluation time in seconds
        :param X: Sets of features
        :param y: Set of labels
        :return: A pipeline or a single estimator, set up with the best parameters
        """
        self._say("Bayesian search with %s iterations..." % self.n_iter)
        for i in range(0, self.n_iter):
            best_params, best_score = self.maximize(current_best_score=self.current_best_score, current_best_time=self.current_best_time)
            self._say("---\nIteration %s/%s. EI: %s" % (i + 1, self.n_iter, best_score))
            self.evaluate(best_params, X, y, current_best_score=self.current_best_score, max_eval_time=max_eval_time)
        return self.get_best()

    def maximize(self, current_best_score, current_best_time):
        """
        Finds the next best point to evaluate.
        :param current_best_score: The current score optimum
        :param current_best_time: The current time optimum
        :return: Parameters of the next point to evaluate
        """
        self.Maximizer.update(self.validated_params, self.validated_scores, self.validated_times, current_best_score, current_best_time)
        return self.Maximizer.maximize()

    def evaluate(self, parameters, X, y, max_eval_time=None, current_best_score=None):
        """
        Evaluates a parameter setting and updates the list of validated parameters afterward.
        :param parameters: The parameter setting to evaluate
        :param X: The sets of features to use for evaluation
        :param y: The set of labels to use for evaluation
        :param max_eval_time: Maximum time to evaluate the parameter setting
        :param current_best_score: The current best score, only used for printing
        :return: A boolean specifying whether or not the evaluation was successful (i.e. finished in time) 
        """
        self._say(
            "Evaluating parameters (timeout: %s s): %s" % (max_eval_time, Converter.readable_parameters(parameters)))

        # Initiate success variable
        success = True

        # Try evaluating within a time limit
        start = time.time()
        try:
            # Build the estimator
            best_estimator = Builder.build_pipeline(self.estimator, parameters)

            # Evaluate with timeout
            with Timeout(max_eval_time):
                score = cross_val_score(best_estimator, X, y, scoring=self.scoring, cv=self.cv, n_jobs=-1)

        except (GeneratorExit, OSError, TimeoutError):
            self._say("Timeout error :(")
            success = False
            score = [self.timeout_score]
        except RuntimeError:
            # It might be that it still works when we set n_jobs to 1 instead of -1.
            # Below we check if we can set n_jobs to 1 and if so, recall this function again.
            if "n_jobs" in self.estimator.get_params() and self.estimator.get_params()["n_jobs"] != 1:
                self._say("Runtime error, trying again with n_jobs=1.")
                self.estimator.set_params(n_jobs=1)
                return self.evaluate(parameters, X, y, max_eval_time, current_best_score)

            # Otherwise, we're going to catch the error as we normally do
            else:
                print("RuntimeError")
                # print(traceback.format_exc())
                success = False
                score = [self.timeout_score]

        except Exception:
            self._say("An error occurred with parameters", Converter.readable_parameters(parameters))
            # print(traceback.format_exc())
            success = False
            score = [self.timeout_score]

        end = time.time() - start if success else max_eval_time

        # Get the mean and store the results
        score = np.mean(score)
        self.validated_scores.append(score)
        self.validated_params.append(parameters)
        self.validated_times.append(end)
        self.current_best_time = min(end, self.current_best_time)
        self.current_best_score = max(score, self.current_best_score)

        if current_best_score is not None:
            self._say("Score: %s | best: %s | time: %s" % (score, max(current_best_score, score), end))
        else:
            self._say("Score: %s | time: %s" % (score, end))

        return success

    def _say(self, *args):
        """
        Calls print() if verbose=True.
        :param args: Arguments to pass to print()
        """
        if self.verbose:
            print(*args)