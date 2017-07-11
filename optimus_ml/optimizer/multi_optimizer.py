import warnings

import numpy as np
from sklearn.model_selection._search import BaseSearchCV
from optimus.vault import decoder

from optimus.extra.fancyprint import say
from optimus.optimizer.builder import Builder
from optimus.optimizer.converter import Converter
from optimus.optimizer.optimizer import Optimizer


class MultiOptimizer(BaseSearchCV):
    def __init__(self, encoded_model_config, scoring="accuracy", n_rounds=50, inner_cv=None, verbose=True,
                 use_ei_per_second=False, max_eval_time=150, n_prep_rounds="auto", max_retries=3, refit=True,
                 timeout_score=0, draw_samples=100):
        """
        An optimizer using Gaussian Processes for optimizing over multiple models and parameter settings. 

        
        Parameters
        ----------
        encoded_model_config: dict with keys `estimator` and `params`
            The `estimator` value is a tuple with 1) the source string of an estimator; and 2) the parameters to 
            initialize the estimator with. The `params` value is a dictionary of parameter distributions for the 
            estimator. An extra key `@preprocessor` can be added to try out different preprocessors. The values of 
            parameters that start with a "!" will be source decoded, and stored under a new key without the prefix.  
            
        scoring: string, callable or None, default=None
            A string (see model evaluation documentation) or a scorer callable object / function with signature
            `scorer(estimator, X, y)`. If `None`, the `score` method of the estimator is used.
        
        n_rounds: int
            Number of iterations for the optimization step
        
        inner_cv: int, cross-validation generator or an iterable, optional
            A scikit-learn compatible cross-validation object
        
        verbose: bool
            Whether or not to print information to the console
        
        use_ei_per_second: bool
            Whether to use the standard EI or the EI / sqrt(second)
            
        max_eval_time: int or float
            Time in seconds until evaluation times out
        
        n_prep_rounds: int
            Number of iterations for each optimizer in the preparation step
        
        max_retries: int
            In case we run into timeouts, how many retries we allow the optimizer to find a setting without a timeout 
            until we remove the optimizer from the list
        
        refit: boolean, default=True
            Refit the best estimator with the entire dataset. If "False", it is impossible to make predictions using
            the estimator instance after fitting
            
        timeout_score: int or float
            The score value to insert in case of timeout
            
        draw_samples: int
            The number of samples to randomly draw from the hyper parameter, to use for finding the next best point.
            
        """

        # Dummy call to super
        super().__init__(None, None, None, None)

        self.encoded_model_config = encoded_model_config
        self.scoring = scoring
        self.n_rounds = n_rounds
        self.inner_cv = inner_cv
        self.verbose = verbose
        self.use_ei_per_second = use_ei_per_second
        self.max_eval_time = max_eval_time
        self.n_prep_rounds = n_prep_rounds
        self.max_retries = max_retries
        self.refit = refit
        self.timeout_score = timeout_score
        self.draw_samples = draw_samples
        self.cv_results_ = None
        self.best_index_ = None
        self.best_estimator_ = None

    def fit(self, X, y):
        """
        Prepare each optimizer for `n_prep_rounds` and then optimize over all optimizers for `n_rounds`.

        Parameters
        ----------
        X: array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples.

        y: array of shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels) as integers or strings.
            
        """

        say("Running MultiOptimizer", self.verbose, style="header")
        self._setup()
        self._prepare(X, y, n_rounds=self.n_prep_rounds, max_retries=self.max_retries)
        self._optimize(X, y, n_rounds=self.n_rounds)

        self.cv_results_, self.best_index_, self.best_estimator_ = self._create_cv_results()

        if self.refit:
            self.best_estimator_.fit(X, y)

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)

    def get_params(self, deep=True):
        return {
            "encoded_model_config": self.encoded_model_config,
            "scoring": self.scoring,
            "n_rounds": self.n_rounds,
            "inner_cv": self.inner_cv,
            "verbose": self.verbose,
            "use_ei_per_second": self.use_ei_per_second,
            "max_eval_time": self.max_eval_time,
            "n_prep_rounds": self.n_prep_rounds,
            "max_retries": self.max_retries
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.__setattr__(parameter, value)
        self._setup()
        return self

    def _setup(self):
        # Setup initial values
        self.optimizers = []
        self.names = []
        self.global_best_score = -np.inf
        self.global_best_time = np.inf
        self.results = {}

        # Setup optimizers
        for cfg in self.encoded_model_config:
            estimator = decoder.decode_source_tuples(cfg["estimator"], special_prefix="!")
            params = decoder.decode_params(cfg["params"])
            optimizer = Optimizer(estimator, param_distributions=params, inner_cv=10,
                                  scoring=self.scoring, timeout_score=self.timeout_score,
                                  max_eval_time=self.max_eval_time, use_ei_per_second=self.use_ei_per_second,
                                  verbose=self.verbose, draw_samples=self.draw_samples)
            self.optimizers.append(optimizer)

    def _prepare(self, X, y, n_rounds="auto", max_retries=3):
        """
        Loop over all optimizers and let each optimizer evaluate a few settings, starting with a random point, but 
        
        Parameters
        ----------
        X: array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples.
            
        y: array of shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels) as integers or strings.
            
        n_rounds: int
            The number of optimization iterations
            
        max_retries: int
            In case we run into timeouts, how many retries we allow the optimizer to find a setting without a timeout 
            until we remove the optimizer from the list

        """

        # Keep a list of indices of model optimizers that could not successfully evaluate their parameters, so that we
        # can remove them later.
        to_remove = []

        for index, optimizer in enumerate(self.optimizers):

            # If n_rounds="auto", adapt the number of rounds to the number of parameters we search over
            num_prep_rounds = n_rounds if n_rounds != "auto" else len(optimizer.param_distributions)

            say("Preparing {} optimizer with {} rounds".format(optimizer, num_prep_rounds), self.verbose, style="title")

            for iteration in range(0, num_prep_rounds):
                say("Iteration {}/{}".format(iteration + 1, num_prep_rounds), self.verbose, style="subtitle")

                # Retry a few times to find a parameter that can be evaluated within max_eval_time.
                success = False
                for i in range(0, max_retries):
                    setting, ei = optimizer.maximize()
                    success, score, running_time = optimizer.evaluate(setting, X, y)
                    self._store_trace(optimizer, setting, score)

                    self.global_best_time = min(self.global_best_time, optimizer.current_best_time)
                    self.global_best_score = max(self.global_best_score, optimizer.current_best_score)

                    if success:
                        break

                # Drop if we never got success
                if not success:
                    to_remove.append(index)
                    break

        # Keep the model optimizers that were successful
        self.optimizers = [i for j, i in enumerate(self.optimizers) if j not in to_remove]

    def _optimize(self, X, y, n_rounds):
        """
        A loop that, in each iteration, compares the highest expected improvement of each optimizer and lets the best 
        optimizer evaluate its setting.
        
        Parameters
        ----------
        X: array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples.
            
        y: array of shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels) as integers or strings.
            
        n_rounds: int
            The number of optimization iterations
            
        """

        say("Optimizing for {} rounds".format(n_rounds), self.verbose, style="title")

        for i in range(0, n_rounds):
            ei, setting, optimizer = self._maximize()

            if optimizer is None:
                warnings.warn("Either all models failed, or you need to call prepare() first.")
                return None

            say("Round {} of {}. Running {} optimizer with EI {}".format(i + 1, n_rounds, optimizer, ei), self.verbose,
                style="subtitle")
            success, score, running_time = optimizer.evaluate(setting, X, y)
            self._store_trace(optimizer, setting, score)

            self.global_best_score = max(self.global_best_score, score)

    def _maximize(self):
        """
        Find optimizer with highest expected improvement.
        
        Returns
        -------
        best_ei: float
            Expected improvement of best optimizer
        
        best_setting: dict
            Best setting found by best optimizer
            
        best_optimizer: Optimizer
            The optimizer with the highest expected improvement
            
        """

        best_ei = -np.inf
        best_setting = None
        best_optimizer = None
        for optimizer in self.optimizers:

            setting, ei = optimizer.maximize(score_optimum=self.global_best_score)

            if ei > best_ei:
                best_ei = ei
                best_setting = setting
                best_optimizer = optimizer

        return best_ei, best_setting, best_optimizer

    def _store_trace(self, optimizer, setting, score):
        """
        Store the optimizer, setting and score, as well as the estimator name and a readable version of the setting.
        
        Parameters
        ----------
        optimizer: Optimizer
            The optimizer used in this round
            
        setting: dict
            The setting used by the optimizer
        
        score: float
            The score found by evaluating the setting
            
        """

        self.results.setdefault("optimizer", []).append(optimizer)
        self.results.setdefault("model_readable", []).append(str(optimizer))
        self.results.setdefault("setting", []).append(setting)
        self.results.setdefault("setting_readable", []).append(Converter.readable_parameters(setting))
        self.results.setdefault("score", []).append(score)

    def _create_cv_results(self):
        """
        Create a slim version of Sklearn's cv_results_ parameter that includes keywords param_model and param_setting,
        which store a readable string of the validated model and the setting, respectively. Next to that, we calculate 
        the best index, and construct the best estimator.

        Returns
        -------
        cv_results: dict of lists
            A table of cross-validation results

        best_index: int
            The index of the best parameter setting

        best_estimator: sklearn estimator
            The estimator initialized with the best parameters

        """

        cv_results = {
            "mean_test_score": self.results["score"],
            "params": self.results["setting"],
            "param_model": self.results["model_readable"],
            "param_setting": self.results["setting_readable"]
        }

        best_index = np.argmax(self.results["score"])  # type: int
        best_setting = self.results["setting"][best_index]
        best_optimizer = self.results["optimizer"][best_index]  # type: Optimizer
        best_estimator = Builder.build_pipeline(best_optimizer.estimator, best_setting)

        return cv_results, best_index, best_estimator
