import time

import numpy as np
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection._search import ParameterSampler, RandomizedSearchCV
from tqdm import tqdm
from optimus_ml.transcoder import converter
from optimus_ml.extra.fancyprint import say
from optimus_ml.optimizer.optimizer import Optimizer


class ModelOptimizer(RandomizedSearchCV):
    def __init__(self, estimator, encoded_params, inner_cv: object = None, scoring="accuracy", timeout_score=0,
                 max_eval_time=120, use_ei_per_second=False, verbose=False, draw_samples=500,
                 n_iter=100, refit=True, random_search=False, time_regression="linear", score_regression="forest",
                 max_run_time=1500, simulate_speedup=1, local_search=False, ls_max_steps=np.inf, multi_start=False,
                 close_neighbors_only=True, xi=0):
        """
        An optimizer using Gaussian Processes for optimizing a single model. 
        
        Parameters
        ----------
        estimator: estimator object
            An object of that type is instantiated for each grid point. This is assumed to implement the scikit-learn 
            estimator interface. Either estimator needs to provide a `score` function, or `scoring` must be passed.
            
        encoded_params: dict
            A dictionary of parameter distributions for the estimator. An extra key `@preprocessor` can be added to try 
            out different preprocessors. The values of parameters that start with a "!" will be source decoded, and 
            stored under a new key without the prefix.  
            
        n_iter: {int, list}
            Number of parameter settings that are drawn using bayesian optimization, when fitting.
            
        draw_samples: int
            The number of samples to randomly draw from the hyper parameter, to use for finding the next best point.
            
        scoring: string, callable or None, default=None
            A string (see model evaluation documentation) or a scorer callable object / function with signature
            `scorer(estimator, X, y)`. If `None`, the `score` method of the estimator is used.
            
        inner_cv: int, cross-validation generator or an iterable, optional
            A scikit-learn compatible cross-validation object
            
        verbose: bool
            Whether or not to print information to the console
            
        timeout_score: int or float
            The score value to insert in case of timeout
             
        use_ei_per_second: bool
            Whether to use the standard EI or the EI / sqrt(second)

        use_root_second: bool
            Only used when "use_ei_per_second=True". Uses EI / sqrt(second) instead of EI /second.

        max_eval_time: int or float
            Time in seconds until evaluation times out
            
        refit: boolean, default=True
            Refit the best estimator with the entire dataset. If "False", it is impossible to make predictions using
            the estimator instance after fitting
            
        random_search: boolean
            If true, use random search instead of bayesian search

        time_regression: str
            Which regression method to use for predicting time

        score_regression: str
            Which regression method to use for the expected improvement

        max_run_time: int
            Maximum running time in seconds

        simulate_speedup: float
            Act as if the time is going slower, for benchmark purposes (e.g. to simulate Randomized 2X)

        local_search: bool
            Whether to do local search

        ls_max_steps: float
            Maximum number of steps to do in local search

        xi: float
            Parameter of EI that controls the exploitation-exploration trade-off
        """

        # Call to super
        super().__init__(None, None, None)

        # Accept parameters
        self.local_search = local_search
        self.ls_max_steps = ls_max_steps
        self.simulate_speedup = simulate_speedup
        self.refit = refit
        self.estimator = estimator
        self.encoded_params = encoded_params
        self.param_distributions = converter.reconstruct_grid(encoded_params)
        self.inner_cv = inner_cv
        self.scoring = scoring
        self.timeout_score = timeout_score
        self.max_eval_time = max_eval_time
        self.use_ei_per_second = use_ei_per_second
        self.verbose = verbose
        self.random_search = random_search
        self.time_regression = time_regression
        self.score_regression = score_regression
        self.max_run_time = max_run_time * self.simulate_speedup
        self.multi_start = multi_start
        self.close_neighbors_only = close_neighbors_only
        self.xi = xi

        # Placeholders for derived variables
        self.draw_samples = draw_samples
        self.n_iter = n_iter
        self.best_estimator_ = None
        self.best_index_ = None
        self.cv_results_ = None
        self.decoded_params = None
        self.optimizer = None
        self.start_time = time.time()

    def fit(self, X, y, skip_setup=False):
        """
        Optimize the model for `n_rounds`.

        Parameters
        ----------
        X: array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples.

        y: array of shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels) as integers or strings.

        skip_setup: bool
            Skip setup. Useful if you need to call setup manually.
        """

        # Calculate derived variables
        if not skip_setup:
            self._setup()

        # Start timer
        self.start_time = time.time()

        # Use Randomized Search or Bayesian Optimization
        if self.random_search:
            self._random_search(X, y, self.n_iter)
        else:
            self._bayesian_search(X, y, self.n_iter)

        # Store results
        self.cv_results_, self.best_index_, self.best_estimator_ = self.optimizer.create_cv_results()
        self.cv_results_["evaluation_time"] = np.array(self.cv_results_["evaluation_time"]) / self.simulate_speedup
        self.cv_results_["maximize_time"] = np.array(self.cv_results_["maximize_time"]) / self.simulate_speedup
        self.cv_results_["total_time"] = np.array(self.cv_results_["total_time"]) / self.simulate_speedup
        self.cv_results_["cumulative_time"] = np.array(self.cv_results_["cumulative_time"]) / self.simulate_speedup

        # Refit the best estimator on the whole dataset
        if self.refit:
            self.best_estimator_.fit(X, y)

        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.__setattr__(parameter, value)
        return self

    def get_params(self, deep=True):
        return {
            "estimator": self.estimator,
            "encoded_params": self.encoded_params,
            "inner_cv": self.inner_cv,
            "scoring": self.scoring,
            "timeout_score": self.timeout_score,
            "max_eval_time": self.max_eval_time,
            "use_ei_per_second": self.use_ei_per_second,
            "verbose": self.verbose,
            "draw_samples": self.draw_samples,
            "n_iter": self.n_iter,
            "random_search": self.random_search
        }

    @staticmethod
    def get_grid_size(param_grid):
        """
        Calculate the grid size (i.e. the number of possible combinations).
        
        Parameters
        ----------
        param_grid: dict
            A dictionary of parameters and their lists of values

        Returns
        -------
        grid_size: int
            The size of the grid.
        """

        grid_size = 1
        for i in param_grid.values():
            grid_size *= len(i)
        return grid_size

    def _bayesian_search(self, X, y, n_iter):
        say("Bayesian search with {} iterations".format(n_iter), self.verbose, style="title")

        for i in tqdm(range(0, n_iter), ascii=True, leave=True):

            # Stop loop if we are out of time
            if self._over_time():
                break

            # Find best setting to evaluate
            if self.multi_start:
                setting, ei = self.optimizer.maximize_multi_start(realize=False)
            else:
                setting, ei = self.optimizer.maximize_single_start(realize=False)

            say("Iteration {}/{}. EI: {}".format(i + 1, n_iter, ei), self.verbose, style="subtitle")

            # Stop loop if we are out of time
            if self._over_time():
                break

            # If we get close to the max_run_time, we set max_eval_time to the remaining time
            self.optimizer.max_eval_time = int(min([self.max_eval_time, self._get_remaining_time()]))

            # Evaluate setting
            self.optimizer.evaluate(setting, X, y)

        # Restore max_eval_time
        self.optimizer.max_eval_time = self.max_eval_time

    def _random_search(self, X, y, n_iter, seed=None):
        # Store random state
        state = np.random.get_state()

        # Set random seed
        if seed is not None:
            np.random.seed(seed)

        say("Randomized search with {} iterations".format(n_iter), self.verbose, style="title")
        samples = [i for i in ParameterSampler(self.decoded_params, n_iter)]

        for i in tqdm(range(0, n_iter), ascii=True, leave=True):

            # Stop loop if we are out of time
            if self._over_time():
                break

            # If we get close to the max_run_time, we set max_eval_time to the remaining time
            self.optimizer.max_eval_time = int(min([self.max_eval_time, self._get_remaining_time()]))

            # Evaluate sample
            setting = samples[i]
            say("Iteration {}/{}.".format(i + 1, n_iter), self.verbose, style="subtitle")
            self.optimizer.evaluate(setting, X, y)

            # Manually add a maximize time of 0, since we don't use the maximize method
            self.optimizer.maximize_times.append(0)

        # Restore random state
        np.random.set_state(state)

        # Restore max_eval_time
        self.optimizer.max_eval_time = self.max_eval_time

    def _over_time(self):
        return self._get_remaining_time() <= 0

    def _get_remaining_time(self):
        return self.max_run_time - (time.time() - self.start_time)

    def _setup(self):

        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        # Set maximum draw samples, and limit the number of iterations for a grid that is too small
        grid_size = self.get_grid_size(self.encoded_params)
        self.draw_samples = min(self.draw_samples, grid_size)
        self.n_iter = min(self.n_iter, grid_size)

        say("Maximum number of iterations as limited by grid: {}".format(grid_size), verbose=self.verbose)

        # Decode parameters for use inside the Model Optimizer
        # self.decoded_params = decoder.decode_params(self.encoded_params)
        self.decoded_params = converter.reconstruct_grid(self.encoded_params)

        # Setup optimizer
        self.optimizer = Optimizer(estimator=self.estimator, param_distributions=self.decoded_params,
                                   inner_cv=self.inner_cv, scoring=self.scoring, timeout_score=self.timeout_score,
                                   max_eval_time=int(self.max_eval_time * self.simulate_speedup),
                                   use_ei_per_second=self.use_ei_per_second,
                                   verbose=self.verbose, draw_samples=self.draw_samples,
                                   time_regression=self.time_regression,
                                   score_regression=self.score_regression,
                                   local_search=self.local_search,
                                   ls_max_steps=self.ls_max_steps,
                                   close_neighbors_only=self.close_neighbors_only,
                                   xi=self.xi)
