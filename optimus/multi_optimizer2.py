import warnings

from sklearn.model_selection._search import BaseSearchCV

from extra.fancyprint import say
from optimus.builder import Builder
from optimus.converter import Converter
from optimus.model_optimizer2 import ModelOptimizer
from optimus.optimizer import Optimizer
from vault import decoder
import numpy as np


class MultiOptimizer(BaseSearchCV):
    def __init__(self, encoded_model_config, scoring="accuracy", n_rounds=50, inner_cv=None, verbose=True,
                 use_ei_per_second=False, max_eval_time=150, n_prep_rounds="auto", max_retries=3, refit=True,
                 timeout_score=0, draw_samples=100):

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

    def fit(self, X, y):
        self._setup()
        self._prepare(X, y, n_rounds=self.n_prep_rounds, max_retries=self.max_retries)
        self._optimize(X, y, n_rounds=self.n_rounds)

    def predict(self, X):
        self.best_estimator_.predict()

    def predict_proba(self, X):
        self.best_estimator_.predict_proba()

    def get_params(self):
        return {
            "model_config": self.encoded_model_config,
            "scoring": self.scoring,
            "n_rounds": self.n_rounds,
            "cv": self.cv,
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
        self.results = {
            "param_model": [],
            "param_setting": []
        }

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
        say("Preparing all models for {} rounds".format(n_rounds), self.verbose, style="header")

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
        say("Optimizing for {} rounds".format(n_rounds), self.verbose, style="header")

        for i in range(0, n_rounds):
            ei, setting, optimizer = self._maximize()

            if optimizer is None:
                warnings.warn("Either all models failed, or you need to call prepare() first.")
                return None

            say("Round {} of {}. Running {} optimizer with EI {}".format(i + 1, n_rounds, optimizer, ei), self.verbose,
                style="title")
            success, score, running_time = optimizer.evaluate(setting, X, y)
            self._store_trace(optimizer, setting, score)

            self.global_best_score = max(self.global_best_score, score)

        self.cv_results_, self.best_index_, self.best_estimator_ = self._create_cv_results()

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
        self.results["model"].append(str(optimizer))
        self.results["setting"].append(Converter.make_readable(setting))
        self.results["score"].append(score)

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
            "param_model": self.results["model"],
            "params": self.results["setting"],
            "param_setting": self.results["setting"]
        }

        best_index = np.argmax(self.results["score"])  # type: int
        best_setting = self.results["params"][best_index]
        best_estimator = Builder.build_pipeline(self.estimator, best_setting)

        return cv_results, best_index, best_estimator
