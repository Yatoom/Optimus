import numpy as np
from sklearn.model_selection._search import BaseSearchCV
from optimus.converter import Converter
from optimus.model_optimizer import ModelOptimizer
from vault import model_factory


class OptimusCV(BaseSearchCV):
    def __init__(self, model_config, n_iter=10, population_size=100, scoring="accuracy", cv=10,
                 verbose=True, timeout_score=0, use_ei_per_second=False, max_eval_time=200):
        # Dummy call to super
        super().__init__(None)

        # Setup model
        self.model_config = model_config
        self.model = model_factory.init_model_config(self.model_config)
        print(self.model_config)
        estimator = self.model["estimator"]
        param_distributions = self.model["params"]

        # Setup optimizer
        self.optimizer = ModelOptimizer(estimator=estimator, param_distributions=param_distributions, n_iter=n_iter,
                                        population_size=population_size, scoring=scoring, cv=cv, verbose=verbose,
                                        timeout_score=timeout_score, use_ei_per_second=use_ei_per_second)
        # Accept parameters
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.population_size = population_size
        self.scoring = scoring
        self.cv = cv
        self.verbose = verbose
        self.timeout_score = timeout_score
        self.use_ei_per_second = use_ei_per_second
        self.max_eval_time = max_eval_time

        # Setup initial parameters
        self.validated_params = []
        self.validated_scores = []
        self.best_estimator_ = None
        self.cv_results_ = None
        self.best_index_ = None

    def fit(self, X, y):
        self.optimizer.fit(X, y, max_eval_time=self.max_eval_time)
        self.validated_params = self.optimizer.validated_params
        self.validated_scores = self.optimizer.validated_scores

        self._store_results(X, y)

        return self

    def _store_results(self, X, y):
        self.best_estimator_ = self.optimizer.get_best().fit(X, y)
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

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)

    def get_params(self, deep=True):
        params = {
            "model_config": self.model_config,
            "n_iter": self.n_iter,
            "population_size": self.population_size,
            "scoring": self.scoring,
            "cv": self.cv,
            "verbose": self.verbose,
            "timeout_score": self.timeout_score,
            "use_ei_per_second": self.use_ei_per_second,
            "max_eval_time": self.max_eval_time
        }
        return params

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.__setattr__(parameter, value)
        return self
