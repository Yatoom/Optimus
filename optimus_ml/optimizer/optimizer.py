import time
import traceback
import warnings

import numpy as np
import copy
import os

from lightgbm import LGBMRegressor

from optimus_ml.extra.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.random_projection import GaussianRandomProjection

if os.name == "nt":
    import pynisher

from optimus_ml.extra.fancyprint import say
from scipy.stats import norm
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, ParameterSampler

from optimus_ml.extra.forests import RandomForestRegressor, ExtraTreesRegressor
from optimus_ml.extra.timeout import Timeout
from optimus_ml.optimizer.builder import Builder
# from optimus_ml.optimizer.converter import Converter
from optimus_ml.transcoder import converter
from smac.epm.rf_with_instances import RandomForestWithInstances

warnings.filterwarnings("ignore")


class Optimizer:
    def __init__(self, estimator, param_distributions, inner_cv=10, scoring="accuracy", timeout_score=0,
                 max_eval_time=120, use_ei_per_second=False, verbose=True, draw_samples=500,
                 time_regression="gp", score_regression="gp", random_state=42, local_search=False,
                 ls_max_steps=np.inf, close_neighbors_only=True, xi=0):
        """
        An optimizer that provides a method to find the next best parameter setting and its expected improvement, and a 
        method to evaluate that parameter setting and keep its results.   
        
        Parameters
        ----------
        estimator : estimator object
            An object of that type is instantiated for each grid point. This is assumed to implement the scikit-learn 
            estimator interface. Either estimator needs to provide a `score` function, or `scoring` must be passed.
            
        param_distributions: dict
            A dictionary of parameter distributions for the estimator. An extra key `@preprocessor` can be added to try 
            out different preprocessors.
        
        inner_cv: int, cross-validation generator or an iterable, optional
            A scikit-learn compatible cross-validation object that will be used for the inner cross-validation
            
        scoring : string, callable or None, default=None
            A string (see model evaluation documentation) or a scorer callable object / function with signature
            `scorer(estimator, X, y)`. If `None`, the `score` method of the estimator is used.
            
        timeout_score: {int, float}
            The score value to insert in case of timeout
            
        max_eval_time: int
            Maximum time for evaluation
            
        use_ei_per_second: bool
            Whether to use the standard EI or the EI / second

        use_root_second: bool
            Only used when "use_ei_per_second=True". Uses EI / sqrt(second) instead of EI /second.
            
        verbose: bool
            Whether to print extra information
            
        draw_samples: int
            Number of randomly selected samples we maximize over

        time_regression: str
            Which classifier to use for predicting running time ("linear": Linear Regression, "gp": Gaussian Processes)

        score_regression: str
            Which classifier to use for predicting running time ("forest": Random Forest, "gp": Gaussian Processes)

        random_state: int
            Random state for the regressors.

        local_search: bool
            Whether to do local search

        ls_max_steps: float
            Maximum number of steps to do in local search

        xi: float
            Parameter of EI that controls the exploitation-exploration trade-off
        """

        # Accept parameters
        self.xi = xi
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.inner_cv = inner_cv
        self.scoring = scoring
        self.timeout_score = timeout_score
        self.max_eval_time = max_eval_time
        self.use_ei_per_second = use_ei_per_second
        self.verbose = verbose
        self.draw_samples = min(draw_samples, self.get_grid_size(param_distributions))
        self.ls_max_steps = ls_max_steps
        self.local_search = local_search

        self.close_neighbors_only = close_neighbors_only

        # Setup initial values
        self.validated_scores = []
        self.validated_params = []
        self.evaluation_times = []
        self.maximize_times = []
        self.cumulative_times = []
        self.best_scores = []
        self.converted_params = None
        self.current_best_score = -np.inf
        self.current_best_time = np.inf

        # Helper function to create Gaussian Process Regressor
        def get_gaussian_process_regressor():
            cov_amplitude = ConstantKernel(1.0, (0.01, 1000.0))
            other_kernel = Matern(
                length_scale=np.ones(len(self.param_distributions)),
                length_scale_bounds=[(0.01, 100)] * len(self.param_distributions),
                nu=2.5)

            return GaussianProcessRegressor(
                kernel=cov_amplitude * other_kernel,
                normalize_y=True, random_state=random_state, alpha=0.0,
                n_restarts_optimizer=2)

        # Helper function to create Random Forest Regressor
        def get_random_forest_regressor():
            return RandomForestRegressor(n_estimators=100, min_samples_leaf=3, min_samples_split=3,
                                         n_jobs=1, max_depth=20, random_state=random_state)

        def get_norm_forest_regressor():
            return make_pipeline(
                Normalizer(norm="max"),
                RandomForestRegressor(bootstrap=False, max_features=0.5, min_samples_leaf=0.05, min_samples_split=0.15,
                                      n_estimators=100, random_state=3)
            )

        # Helper function to create Extra Trees Regressor
        def get_extra_trees_regressor():
            return ExtraTreesRegressor(n_estimators=100, min_samples_leaf=3, min_samples_split=3,
                                       n_jobs=1, max_depth=20, random_state=random_state)

        def get_adaboost_regressor():
            return AdaBoostRegressor(n_estimators=50, random_state=random_state)

        # Helper function to create Linear Regressor
        def get_linear_regressor():
            return LinearRegression(normalize=True, n_jobs=1)

        def get_gradient_boosting_regressor():
            return GradientBoostingRegressor(n_estimators=100, min_samples_leaf=3, min_samples_split=3, max_depth=20,
                                             random_state=random_state)

        # Setup score regressor (for predicting EI)
        self.score_regression = score_regression
        if score_regression == "forest":
            self.score_regressor = get_random_forest_regressor()
        elif score_regression == "extra forest":
            self.score_regressor = get_extra_trees_regressor()
        elif score_regression == "gp":
            self.score_regressor = get_gaussian_process_regressor()
        elif score_regression == "normal forest":
            self.score_regressor = get_norm_forest_regressor()
        elif score_regression == "SMAC-forest":
            params = self.param_distributions
            bounds = np.array([[0.0, float(len(j))] for i, j in params.items()])
            types = np.array([0 for _ in params])
            self.score_regressor = RandomForestWithInstances(bounds=bounds, types=types)
        else:
            raise ValueError("The value '{}' is not a valid value for 'score_regression'".format(score_regression))

        # Setup score regressor (for predicting running time)
        if use_ei_per_second:
            if time_regression == "forest":
                self.time_regressor = get_random_forest_regressor()
            elif time_regression == "extra forest":
                self.time_regressor = get_extra_trees_regressor()
            elif time_regression == "gp":
                self.time_regressor = get_gaussian_process_regressor()
            elif time_regression == "linear":
                self.time_regressor = get_linear_regressor()
            elif time_regression == "gradient":
                self.time_regressor = get_gradient_boosting_regressor()
            elif time_regression == "adaboost":
                self.time_regressor = get_adaboost_regressor()
            elif time_regression == "normal forest":
                self.time_regressor = get_norm_forest_regressor()
            elif time_regression == "lightgbm":
                self.time_regressor = LGBMRegressor(max_depth=3, objective="rmse", verbose=-1, n_estimators=100, min_child_samples=1)
            else:
                raise ValueError("The value '{}' is not a valid value for 'time_regression'".format(time_regression))

    def __str__(self):
        # Returns the name of the estimator (e.g. LogisticRegression)
        return type(self.estimator).__name__

    def maximize_multi_start(self, score_optimum=None, realize=False):
        start = time.time()

        # Select a sample of parameters
        sampled_params = ParameterSampler(self.param_distributions, self.draw_samples)

        # Set score optimum
        if score_optimum is None:
            score_optimum = self.current_best_score

        # # Determine the best parameters
        # best_setting, best_score = self._maximize_on_sample(sampled_params, score_optimum)

        best_score = -np.inf
        best_setting = None

        for setting in sampled_params:
            setting, score = self._local_search(setting, -np.inf, score_optimum,
                                                max_steps=self.ls_max_steps)
            if score > best_score:
                best_score = score
                best_setting = setting

        if realize:
            best_setting, best_score = self._realize(best_setting, best_score, score_optimum)

        # Store running time
        running_time = time.time() - start
        self.maximize_times.append(running_time)

        return best_setting, best_score

    def maximize_single_start(self, score_optimum=None, realize=False):
        """
        Find the next best hyper-parameter setting to optimizer.

        Parameters
        ----------
        score_optimum: float
            An optional score to use inside the EI formula instead of the optimizer's current_best_score

        realize: bool
            Whether or not to give a more realistic estimate of the EI (default=True)

        Returns
        -------
        best_setting: dict
            The setting with the highest expected improvement
        
        best_score: float
            The highest EI (per second)
        """

        start = time.time()

        # Select a sample of parameters
        sampled_params = ParameterSampler(self.param_distributions, self.draw_samples)

        # Set score optimum
        if score_optimum is None:
            score_optimum = self.current_best_score

        # Determine the best parameters
        best_setting, best_score = self._maximize_on_sample(sampled_params, score_optimum)

        if self.local_search:
            # max_steps = int(len(self.validated_scores) / 50)
            max_steps = np.inf
            best_setting, best_score = self._local_search(best_setting, best_score, score_optimum,
                                                          max_steps=max_steps)

        if realize:
            best_setting, best_score = self._realize(best_setting, best_score, score_optimum)

        # Store running time
        running_time = time.time() - start
        self.maximize_times.append(running_time)

        return best_setting, best_score

    def evaluate(self, parameters, X, y):
        """
        Evaluates a parameter setting and updates the list of validated parameters afterward.
        
        Parameters
        ----------
        parameters: dict
            The parameter settings to evaluate
            
        X: array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples
            
        y: array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels) as integers or strings
                       
        Returns
        -------
        success: bool
            Whether or not the evaluation was successful (i.e. finished in time)
        
        score: float
            The resulting score (equals timeout_score if evaluation was not successful)
            
        running_time: float
            The running time in seconds (equals max_eval_time if evaluation was not successful)        
        """
        say("Evaluating parameters (timeout: %s s): %s" % (
            self.max_eval_time, converter.dictionary_to_string(parameters)), self.verbose)

        # Initiate success variable
        success = True

        # Try evaluating within a time limit
        start = time.time()
        try:
            # Build the estimator
            best_estimator = Builder.build_pipeline(self.estimator, parameters)

            # Evaluate with timeout (Windows)
            if os.name == "nt":
                timed_cross_val_score = pynisher.enforce_limits(
                    wall_time_in_s=self.max_eval_time)(cross_val_score)
                score = timed_cross_val_score(estimator=best_estimator, X=X, y=y, scoring=self.scoring,
                                              cv=self.inner_cv, n_jobs=-1)

                if timed_cross_val_score.exit_status != 0:
                    raise TimeoutError

            # Evaluate with timeout (Linux)
            else:
                with Timeout(self.max_eval_time):
                    score = cross_val_score(estimator=best_estimator, X=X, y=y, scoring=self.scoring, cv=self.inner_cv,
                                            n_jobs=-1)

        except (GeneratorExit, OSError, TimeoutError, BrokenPipeError, ImportError):
            say("Timeout error :(", self.verbose)
            success = False
            score = [self.timeout_score]
        except RuntimeError:
            # It might be that it still works when we set n_jobs to 1 instead of -1.
            # Below we check if we can set n_jobs to 1 and if so, recall this function again.
            if "n_jobs" in self.estimator.get_params() and self.estimator.get_params()["n_jobs"] != 1:
                say("Runtime error, trying again with n_jobs=1.", self.verbose)
                self.estimator.set_params(n_jobs=1)
                return self.evaluate(parameters, X, y)

            # Otherwise, we're going to catch the error as we normally do
            else:
                say("RuntimeError", self.verbose)
                print(traceback.format_exc())
                success = False
                score = [self.timeout_score]

        except Exception:
            say("An error occurred with parameters {}".format(converter.dictionary_to_string(parameters)), self.verbose)
            print(traceback.format_exc())
            success = False
            score = [self.timeout_score]

        # Get the mean and store the results
        score = np.mean(score)  # type: float
        self.validated_scores.append(score)
        self.validated_params.append(parameters)
        self.converted_params = converter.settings_to_indices(self.validated_params, self.param_distributions)

        self.current_best_score = max(score, self.current_best_score)
        self.best_scores.append(self.current_best_score)

        running_time = time.time() - start if success else self.max_eval_time
        self.evaluation_times.append(running_time)
        self.current_best_time = min(running_time, self.current_best_time)

        say("Score: %s | best: %s | time: %s" % (score, self.current_best_score, running_time), self.verbose)
        return success, score, running_time

    def create_cv_results(self):
        """
        Create a slim version of Sklearn's cv_results_ parameter that includes the keywords "params", "param_*" and 
        "mean_test_score", calculate the best index, and construct the best estimator.

        Returns
        -------
        cv_results : dict of lists
            A table of cross-validation results

        best_index: int
            The index of the best parameter setting

        best_estimator: sklearn estimator
            The estimator initialized with the best parameters

        """

        # Cleanup: make sure the number of maximize times equals the number of evaluation times. It could be that there
        # are more maximize times, when the model optimizer runs out of time and can't run a matching evaluate() for the
        # last maximize() call. On the other hand, when we do a random search, we only need to evaluate(), so if we
        # forget to manually add a maximize time of 0, we will fill everything up with zeros here.
        maximize_times = self.maximize_times
        while len(maximize_times) > len(self.evaluation_times):
            maximize_times.pop()
        while len(self.evaluation_times) > len(maximize_times):
            maximize_times.append(0)

        cv_results = {
            "params": self.validated_params,
            "readable_params": [converter.dictionary_to_readable_dictionary(dict_) for dict_ in self.validated_params],
            "mean_test_score": self.validated_scores,
            "evaluation_time": self.evaluation_times,
            "maximize_time": maximize_times,
            "best_score": self.best_scores,
            "total_time": maximize_times + self.evaluation_times,
            "cumulative_time": np.cumsum(maximize_times) + np.cumsum(self.evaluation_times)
        }

        # Insert "param_*" keywords
        for setting in cv_results["params"]:
            for key, item in setting.items():
                param = "param_{}".format(key)

                # Create keyword if it does not exist
                if param not in cv_results:
                    cv_results[param] = []

                # Use cleaner names
                # TODO: make this reproducible from OpenML
                value = converter.value_to_json(item, openml_compatible=True)

                # Add value to results
                cv_results[param].append(value)

        # Find best index
        best_index = np.argmax(self.validated_scores)  # type: int
        best_setting = self.validated_params[best_index]
        best_estimator = Builder.build_pipeline(self.estimator, best_setting)

        return cv_results, best_index, best_estimator

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

    def _maximize_on_sample(self, sampled_params, score_optimum):
        """
        Finds the next best setting to evaluate from a set of samples. 
        
        Parameters
        ----------
        sampled_params: list
            The samples to calculate the expected improvement on
            
        score_optimum: float
            The score optimum value to pass to the EI formula

        Returns
        -------
        best_setting: dict
            The setting with the highest expected improvement
        
        best_score: float
            The highest EI (per second)
        """

        # A little trick to count the number of validated scores that are not equal to the timeout_score value
        # Numpy's count_nonzero is used to count non-False's instead of non-zeros.
        num_valid_scores = np.count_nonzero(~(np.array(self.validated_scores) == self.timeout_score))

        # Convert sampled_params to list and remove already tried parameters
        sampled_params_list = [i for i in sampled_params if i not in self.validated_params]

        # Check if the number of validated scores (without timeouts) is zero
        if num_valid_scores == 0:
            return np.random.choice(sampled_params_list), 0

        self._fit(self.converted_params, self.validated_scores, self.evaluation_times, remove_timeouts=False)

        converted_settings = converter.settings_to_indices(sampled_params_list,
                                                           param_distributions=self.param_distributions)
        scores = self._get_eis_per_second(converted_settings, score_optimum)
        best_index = np.argmax(scores)  # type: int
        best_score = scores[best_index]
        best_setting = sampled_params_list[best_index]
        return best_setting, best_score

    def _local_search(self, best_setting, best_score, score_optimum, max_steps=np.inf):
        n_steps = 0

        while n_steps <= max_steps:
            neighbors = self._get_neighborhood(best_setting)
            new_setting, new_score = self._maximize_on_sample(neighbors, score_optimum)

            if new_score > best_score:
                best_score = new_score
                best_setting = new_setting
            else:
                break

            n_steps += 1

        return best_setting, best_score

    def _get_neighborhood(self, setting):
        if self.close_neighbors_only:
            return self._get_neighbors(setting)

        neighbors = []
        params = self.param_distributions

        for key, value in setting.items():
            possible_values = params[key]
            neighbor = copy.copy(setting)

            for val in possible_values:
                if val == value:
                    continue
                neighbor[key] = val
                neighbors.append(copy.copy(neighbor))

        return neighbors

    def _get_neighbors(self, setting):
        neighbors = []
        params = self.param_distributions

        for key, value in setting.items():
            possible_values = params[key]
            position_self = possible_values.index(value)

            neighbor = copy.copy(setting)
            if position_self - 1 >= 0:
                neighbor[key] = possible_values[position_self - 1]
                neighbors.append(copy.copy(neighbor))
            if position_self + 1 < len(possible_values):
                neighbor[key] = possible_values[position_self + 1]
                neighbors.append(copy.copy(neighbor))

        return neighbors

    def _realize(self, best_setting, original, score_optimum):
        """
        Calculate a more realistic estimate of the expected improvement by removing validations that resulted in a 
        timeout. These timeout scores are useful to direct the Gaussian Process away, but if we need a realistic 
        estimation of the expected improvement, we should remove these points.
        
        Parameters
        ----------
        best_setting: dict
            The setting to calculate a realistic estimate for
            
        original: float
            The original estimate, which will be returned in case we can not calculate the realistic estimate
            
        score_optimum: float
            The score optimum value to pass to the EI formula

        Returns
        -------
        Returns the realistic estimate
        """

    # def _get_ei_per_second(self, point, score_optimum):
    #     """
    #     Wrapper for _get_eis_per_second()
    #     """
    #     eis = self._get_eis_per_second([point], score_optimum)
    #     return eis[0]

    def _get_eis_per_second(self, points, score_optimum):
        """
        Calculate the expected improvement and divide it by the (square root) of the running time, if
        "self.use_ei_per_second == True".

        Parameters
        ----------
        points:
            Settings to predict on

        score_optimum: float
            The score optimum value to use inside the EI formula

        Returns
        -------
        Return the Expected Improvements (per (root) second)
        """

        eis = self._get_eis(points, score_optimum).reshape(-1)

        if self.use_ei_per_second:

            # Predict running times
            running_times = self._predict_time(points)

            # Some algorithms, such as Linear Regression, predict negative values for the running time. In that case,
            # we take the lowest evaluation time we have observed so far
            # for index, value in enumerate(running_times):
            #     if value <= 0:
            #         running_times[index] = np.min(self.evaluation_times)

            return eis / np.log(running_times + 1)

        return eis

    def _get_eis(self, points, score_optimum):
        """
        Calculate the expected improvements for all points.

        Parameters
        ----------
        points: list
            List of parameter settings for the GP to predict on

        score_optimum: float
            The score optimum value to use for calculating the difference against the expected value

        Returns
        -------
        Returns the Expected Improvement
        """

        # Predict mu's and sigmas for each point
        mu, sigma = self._predict_score(points)

        # Subtract each item in list by score_optimum
        # We subtract 0.01 because http://haikufactory.com/files/bayopt.pdf
        # (2.3.2 Exploration-exploitation trade-of)
        diff = mu - score_optimum - self.xi

        # Divide each diff by each sigma
        Z = diff / sigma

        # Calculate EI's
        ei = diff * norm.cdf(Z) + sigma * norm.pdf(Z)

        # Make EI zero when sigma is zero (but then -1 when sigma <= 1e-05 to be more sure that everything goes well)
        for index, value in enumerate(sigma):
            if value <= 1e-05:
                ei[index] = -1

        return ei

    def _fit(self, params, scores, times, remove_timeouts=False):
        params_array = np.array(params).astype(float)
        scores_array = np.array(scores).astype(float)

        # Remove timeouts if desired
        if remove_timeouts:
            params_array, scores_array = converter.remove_timeouts(params_array, scores_array, self.timeout_score)

        # Try to fit score regressor (and time regressor)
        try:
            if self.score_regression == "SMAC-forest":
                self.score_regressor.train(params_array, scores_array)
            else:
                self.score_regressor.fit(params_array, scores_array)

            if self.use_ei_per_second:
                self.time_regressor.fit(params_array, scores_array)
        except:
            print(traceback.format_exc())

    def _predict_score(self, points):
        points_array = np.array(points).astype(float)

        if self.score_regression == "SMAC-forest":
            mu, var = self.score_regressor.predict(points_array)
            mu, sigma = mu, np.sqrt(var)
        else:
            mu, sigma = self.score_regressor.predict(points_array, return_std=True)
        return mu, sigma

    def _predict_time(self, points):
        running_times = self.time_regressor.predict(points)
        return running_times
