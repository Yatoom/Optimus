import warnings

from sklearn.model_selection._search import BaseSearchCV
from vault import decoder
from optimus.model_optimizer import ModelOptimizer
import numpy as np
import pandas as pd


class MultiOptimizer(BaseSearchCV):
    def __init__(self, model_config, scoring="accuracy", n_rounds=50, cv=10, verbose=True, use_ei_per_second=False,
                 max_eval_time=150, n_prep_rounds="auto", max_retries=3):
        """
        Optimizer for multiple models.
        :param models: Dictionary with a 'name': string, an 'estimator': Sklearn Estimator and 'params': dictionary.
        :param scoring: A Sklearn scorer: http://scikit-learn.org/stable/modules/model_evaluation.html
        :param cv: A Sklearn Cross-Validation object or number
        :param verbose: A boolean to control verbosity
        :param use_ei_per_second: Whether to use the standard EI or the EI / sqrt(second)
        """

        # Dummy call to super
        super().__init__(None, None, None, None)

        # Accept parameters
        self.model_config = model_config
        self.scoring = scoring
        self.n_rounds = n_rounds
        self.cv = cv
        self.verbose = verbose
        self.use_ei_per_second = use_ei_per_second
        self.max_eval_time = max_eval_time
        self.n_prep_rounds = n_prep_rounds
        self.max_retries = max_retries

        self.model_optimizers = self.names = self.global_best_score = self.global_best_time = self.results = None
        self.best_estimator_ = None
        self.cv_results_ = None
        self.setup()

    def setup(self):
        # Setup initial values
        self.model_optimizers = []
        self.names = []
        self.global_best_score = -np.inf
        self.global_best_time = np.inf
        self.results = []

        # Setup optimizers
        for cfg in self.model_config:
            estimator = decoder.decode_sources(cfg["estimator"], special_prefix="!")
            params = cfg["params"]
            model_optimizer = ModelOptimizer(estimator, params, n_iter=None, population_size=100, scoring=self.scoring,
                                             cv=self.cv, verbose=self.verbose, use_ei_per_second=self.use_ei_per_second,
                                             max_eval_time=self.max_eval_time)
            self.model_optimizers.append(model_optimizer)

    def prepare(self, X, y, n_rounds="auto", max_retries=3):
        self._say("Preparing all models for %s rounds" % n_rounds, "-")

        # Keep a list of indices of model optimizers that could not successfully evaluate their parameters, so that we
        # can remove them later.
        to_remove = []

        for index, optimizer in enumerate(self.model_optimizers):

            num_params = len(optimizer.param_distributions)
            num_prep_rounds = n_rounds if n_rounds != "auto" else num_params
            self._say("Preparing {} optimizer with {} rounds".format(optimizer, num_prep_rounds))

            for iteration in range(0, num_prep_rounds):
                self._say("Iteration {}/{}".format(iteration + 1, num_prep_rounds), "-", "-")

                # Retry a few times to find a parameter that can be evaluated within max_eval_time.
                success = False
                for i in range(0, max_retries):
                    parameters, score = optimizer.maximize(self.global_best_score, self.global_best_time)
                    success = optimizer.evaluate(parameters, X, y, current_best_score=self.global_best_score)

                    if success:
                        break

                # Drop if we never got success
                if not success:
                    to_remove.append(index)
                    break

                # Update global best score
                self.global_best_time = min(self.global_best_time, optimizer.current_best_time)
                self.global_best_score = max(self.global_best_score, optimizer.current_best_score)

        # Keep the model optimizers that were successful
        self.model_optimizers = [i for j, i in enumerate(self.model_optimizers) if j not in to_remove]

        # Store results so far
        self._store_results(X, y)

    def optimize(self, X, y, n_rounds):
        self._say("Optimizing for {} rounds.".format(n_rounds), "-")

        for i in range(0, n_rounds):
            best_optimizer, best_parameters, best_score = self._get_best_ei()

            if best_optimizer is None:
                warnings.warn("Either all models failed, or you need to call prepare() first.")
                return None

            self._say("Round {} of {}. Running {} optimizer with EI {}".format(i + 1, n_rounds, best_optimizer,
                                                                               best_score))
            best_optimizer.evaluate(best_parameters, X, y)
            self.global_best_score = max(self.global_best_score, best_optimizer.current_best_score)

        # Store results
        self._store_results(X, y)

    def fit(self, X, y):
        self.prepare(X, y, n_rounds=self.n_prep_rounds, max_retries=self.max_retries)
        self.optimize(X, y, n_rounds=self.n_rounds)
        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)

    def _store_results(self, X, y):

        # Get the best estimator
        self.best_estimator_ = None
        best_score = -np.inf
        for optimizer in self.model_optimizers:
            if optimizer.current_best_score > best_score:
                best_score = optimizer.current_best_score
                self.best_estimator_ = optimizer.best_estimator_

        # Refit the best estimator
        if self.refit:
            self.best_estimator_.fit(X, y)

        # Get all results with prefixed params
        results = []
        for optimizer in self.model_optimizers:
            results.append(pd.DataFrame(optimizer.get_results(prefix=optimizer)))

        # Merge results together
        merged = pd.concat(results, axis=0, ignore_index=True)
        merged = merged.fillna("N/A")

        # Store results
        self.best_index_ = np.argmax(merged["mean_test_score"])
        self.cv_results_ = merged.to_dict()

    def _get_best_ei(self):
        """
        Ask each individual model optimizer to calculate its expected improvement and get the model with the best EI and
        the corresponding parameters. 
        
        Returns
        -------
        best_optimizer: ModelOptimizer
            The ModelOptimizer that found the best (highest EI) setting        
        best_parameters: dict
            The parameters that the best ModelOptimizer found
        best_score: float
            The highest EI that corresponds to the best optimizer's parameters
        """

        best_parameters = None
        best_optimizer = None
        best_score = -np.inf
        for optimizer in self.model_optimizers:
            parameters, score = optimizer.maximize(self.global_best_score, self.global_best_time)

            if score > best_score:
                best_parameters = parameters
                best_optimizer = optimizer
                best_score = score

        return best_optimizer, best_parameters, best_score

    def _reset(self):
        self.global_best_score = -np.inf
        self.global_best_time = np.inf
        for optimizer in self.model_optimizers:
            optimizer.validated_params = []
            optimizer.validated_scores = []
            optimizer.validated_times = []
            optimizer.current_best_setting = None
            optimizer.current_best_score = -np.inf
            optimizer.current_best_time = np.inf

    def get_params(self, deep=True):
        return {
            "model_config": self.model_config,
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
        self.setup()
        return self

    def _say(self, output, underline=None, separator=None):
        if self.verbose:

            # Upper line
            if separator is not None:
                print("\n" + str(separator * len(output)))

            # Output
            print(output)

            # Under line
            if underline is not None:
                print(str(underline * len(output)) + "\n")