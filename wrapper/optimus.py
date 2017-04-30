from sklearn.model_selection._search import BaseSearchCV
import numpy as np
from prime.multi_optimizer import MultiOptimizer
from vault import models


class Optimus(BaseSearchCV):
    def __init__(self, scoring="accuracy", cv=10, verbose=True, use_ei_per_second=False, prep_rounds="auto",
                 opt_rounds=50, max_eval_time=150, max_prep_retries=4, categorical=None, missing=False):

        # Get models
        if categorical is None:
            categorical = []
        self.categorical = categorical
        self.missing = missing
        self.models = models.get_models(categorical, missing, random_state=3)

        # Setup optimizer
        self.scoring = scoring
        self.cv = cv
        self.verbose = verbose
        self.use_ei_per_second = use_ei_per_second
        self.optimizer = MultiOptimizer(self.models, scoring, cv, verbose, use_ei_per_second=use_ei_per_second)

        # Accept other parameters
        self.prep_rounds = prep_rounds
        self.max_eval_time = max_eval_time
        self.opt_rounds = opt_rounds
        self.max_prep_retries = max_prep_retries

        # Setup initial values
        self.best = None
        self.results = None
        self.best_estimator_ = None

        # We don't actually want to use the BaseSearchCV, we just need to be an instance of it for compatibility with
        # OpenML. So we don't need to send actual stuff to the super class.
        super().__init__(None, None, None, None)

    @property
    def cv_results_(self):
        return {
            "mean_test_score": [item["score"] for item in self.results],
            "param_model": [item["model"] for item in self.results],
        }

    @property
    def best_index_(self):
        return np.argmax([item["score"] for item in self.results])

    def fit(self, X, y):
        self.optimizer.reset()
        print("Preparing\n---------")
        self.optimizer.prepare(X, y, n_rounds=self.prep_rounds, max_eval_time=self.max_eval_time,
                               max_retries=self.max_prep_retries)
        print("Optimizing\n----------")
        best, name, score = self.optimizer.optimize(X, y, n_rounds=self.opt_rounds, max_eval_time=self.max_eval_time)
        # results = self.optimizer.get_all_validated_params()
        results = self.optimizer.results

        self.best = best
        self.results = results

        self.best_estimator_ = best.fit(X, y)

        return self

    def predict(self, X):
        return self.best.predict(X)

    def predict_proba(self, X):
        return self.best.predict_proba(X)

    def get_params(self, deep=False):
        params = {
            "scoring": self.scoring,
            "cv": self.cv,
            "verbose": self.verbose,
            "use_ei_per_second": self.use_ei_per_second,
            "prep_rounds": self.prep_rounds,
            "max_eval_time": self.max_eval_time,
            "opt_rounds": self.opt_rounds,
            "max_prep_retries": self.max_prep_retries,
            "categorical": self.categorical,
            "missing": self.missing
        }
        return params

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.__setattr__(parameter, value)
        return self
