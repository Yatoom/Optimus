import threading
import time
import traceback

from sklearn.model_selection import cross_val_score
import numpy as np
import warnings

from sklearn.model_selection._search import BaseSearchCV

from extra.timeout import Timeout
from optimus.builder import Builder
from optimus.converter import Converter
from optimus.maximizer import Maximizer
from vault import decoder

warnings.filterwarnings("ignore")


class ModelOptimizer(BaseSearchCV):
    def __init__(self, estimator, param_distributions, n_iter=10, population_size=100, scoring=None, cv=10,
                 verbose=True, timeout_score=0, use_ei_per_second=False, max_eval_time=None, refit=True):
        """
        An optimizer using Gaussian Processes for optimizing a single model. 
        
        Parameters
        ----------
        estimator : estimator object
            An object of that type is instantiated for each grid point. This is assumed to implement the scikit-learn 
            estimator interface. Either estimator needs to provide a `score` function, or `scoring` must be passed.
            
        param_distributions: dict
            A dictionary of parameter distributions for the estimator. An extra key `@preprocessor` can be added to try 
            out different preprocessors. The values of parameters that start with a "!" will be source decoded, and 
            stored under a new key without the prefix.  
            
        n_iter:
            Number of parameter settings that are drawn using bayesian optimization, when fitting.
            
        population_size: int
            The number of samples to randomly draw from the hyper parameter, to use for finding the next best point.
            
        scoring : string, callable or None, default=None
            A string (see model evaluation documentation) or a scorer callable object / function with signature
            `scorer(estimator, X, y)`. If `None`, the `score` method of the estimator is used.
            
        cv: int, cross-validation generator or an iterable, optional
            A scikit-learn compatible cross-validation object
            
        verbose: bool
            Whether or not to print information to the console
            
        timeout_score: {int, float}
            The score value to insert in case of timeout
             
        use_ei_per_second: bool
            Whether to use the standard EI or the EI / sqrt(second)
            
        max_eval_time: 
            Maximum time for evaluation
            
        refit : boolean, default=True
            Refit the best estimator with the entire dataset. If "False", it is impossible to make predictions using
            the estimator instance after fitting
        """

        # Dummy call to super
        super().__init__(None, None, None, None)

        # Accept parameters
        self.refit = refit
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.max_eval_time = max_eval_time
        self.n_iter = n_iter
        self.population_size = population_size
        self.scoring = scoring
        self.cv = cv
        self.verbose = verbose
        self.timeout_score = timeout_score
        self.use_ei_per_second = use_ei_per_second

        # Initialization of validated values, scores and the current best setting
        self.validated_params = []
        self.validated_scores = []
        self.validated_times = []
        self.current_best_setting = None
        self.current_best_score = -np.inf
        self.current_best_time = np.inf
        self.best_estimator_ = None

        # Accept either a normal or a to-be source-decoded parameter grid
        self.decoded_grid = decoder.decode_params(param_distributions, prefix="!", remove_prefixes=True)

        # Setup maximizer
        self.Maximizer = Maximizer(param_distribution=self.decoded_grid, timeout_score=self.timeout_score,
                                   use_ei_per_second=use_ei_per_second)

    def setup(self):
        # Decode grid
        self.decoded_grid = decoder.decode_params(self.param_distributions, prefix="!", remove_prefixes=True)

        # Initialization of validated values, scores and the current best setting
        self.validated_params = []
        self.validated_scores = []
        self.validated_times = []
        self.current_best_setting = None
        self.current_best_score = -np.inf
        self.current_best_time = np.inf
        self.best_estimator_ = None

        # Setup maximizer
        self.Maximizer = Maximizer(param_distribution=self.decoded_grid, timeout_score=self.timeout_score,
                                   use_ei_per_second=self.use_ei_per_second)

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

    def fit(self, X, y):
        """
        Fit method for stand-alone use of the Optimus single-model optimizer.
        :param max_eval_time: Maximum evaluation time in seconds
        :param X: Sets of features
        :param y: Set of labels
        :return: A pipeline or a single estimator, set up with the best parameters
        """

        # Limit the number of iterations for a grid that is too small
        self.n_iter = min(Maximizer.get_grid_size(self.decoded_grid), self.n_iter)
        self._say("Bayesian search with %s iterations..." % self.n_iter)

        for i in range(0, self.n_iter):
            best_params, best_score = self.maximize(current_best_score=self.current_best_score,
                                                    current_best_time=self.current_best_time)
            self._say("---\nIteration %s/%s. EI: %s" % (i + 1, self.n_iter, best_score))
            self.evaluate(best_params, X, y, current_best_score=self.current_best_score)

        self._store_results(X, y)
        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)

    def maximize(self, current_best_score, current_best_time):
        """
        Finds the next best point to evaluate.
        :param current_best_score: The current score optimum
        :param current_best_time: The current time optimum
        :return: Parameters of the next point to evaluate
        """
        self.Maximizer.update(self.validated_params, self.validated_scores, self.validated_times, current_best_score,
                              current_best_time)
        return self.Maximizer.maximize()

    def evaluate(self, parameters, X, y, current_best_score=None):
        """
        Evaluates a parameter setting and updates the list of validated parameters afterward.
        :param parameters: The parameter setting to evaluate
        :param X: The sets of features to use for evaluation
        :param y: The set of labels to use for evaluation
        :param current_best_score: The current best score, only used for printing
        :return: A boolean specifying whether or not the evaluation was successful (i.e. finished in time) 
        """
        self._say(
            "Evaluating parameters (timeout: %s s): %s" % (
            self.max_eval_time, Converter.readable_parameters(parameters)))

        # Initiate success variable
        success = True

        # Try evaluating within a time limit
        start = time.time()
        try:
            # Build the estimator
            best_estimator = Builder.build_pipeline(self.estimator, parameters)

            # Evaluate with timeout
            with Timeout(self.max_eval_time):
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
                return self.evaluate(parameters, X, y, current_best_score)

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

        end = time.time() - start if success else self.max_eval_time

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

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.__setattr__(parameter, value)
        self.setup()
        return self

    def get_params(self, deep=True):
        return {
            "estimator": self.estimator,
            "param_distributions": self.param_distributions,
            "n_iter": self.n_iter,
            "population_size": self.population_size,
            "scoring": self.scoring,
            "cv": self.cv,
            "verbose": self.verbose,
            "timeout_score": self.timeout_score,
            "use_ei_per_second": self.use_ei_per_second,
            "max_eval_time": self.max_eval_time,
            "refit": self.refit
        }

    def _say(self, *args):
        """
        Calls print() if verbose=True.
        :param args: Arguments to pass to print()
        """
        if self.verbose:
            print(*args)

    def get_results(self, prefix=None):
        if prefix is None:
            return self.cv_results_

        prefixed = {
            "mean_test_score": self.validated_scores,
            "params": self.validated_params
        }
        for key, value in self.cv_results_.items():
            if key.startswith("param_"):
                prefixed["{}__param_{}".format(prefix, key)] = value

        return prefixed

    def _store_results(self, X, y):
        self.best_estimator_ = self.get_best()
        if self.refit:
            self.best_estimator_.fit(X, y)
        self.cv_results_ = {
            "mean_test_score": self.validated_scores,
            "params": self.validated_params
        }

        self.best_index_ = np.argmax(self.cv_results_["mean_test_score"])

        for key in self.validated_params[0]:
            self.cv_results_["param_%s" % key] = []

        for setting in self.validated_params:
            for key, value in setting.items():
                value = Converter.make_readable(value)
                self.cv_results_["param_%s" % key].append(value)
