from optimus.builder import Builder
from sklearn.model_selection._search import BaseSearchCV
from optimus.optimizer import Optimizer
from extra.fancyprint import say
from vault import decoder
import numpy as np


class ModelOptimizer(BaseSearchCV):
    def __init__(self, estimator, encoded_params, inner_cv=10, scoring="accuracy", timeout_score=0, max_eval_time=120,
                 use_ei_per_second=False, verbose=True, draw_samples=100, n_iter=10, refit=True):
        # Call to super
        super().__init__(None, None, None)

        # Accept parameters
        self.refit = refit
        self.estimator = estimator
        self.encoded_params = encoded_params
        self.inner_cv = inner_cv
        self.scoring = scoring
        self.timeout_score = timeout_score
        self.max_eval_time = max_eval_time
        self.use_ei_per_second = use_ei_per_second
        self.verbose = verbose

        # Placeholders for derived variables
        self.draw_samples = draw_samples
        self.n_iter = n_iter
        self.best_estimator_ = None
        self.best_index_ = None
        self.cv_results_ = None
        self.decoded_params = None
        self.optimizer = None

        # Calculate derived variables
        self._setup()

    def fit(self, X, y):

        say("Bayesian search with %s iterations" % self.n_iter, self.verbose, style="title")

        for i in range(0, self.n_iter):
            setting, ei = self.optimizer.maximize()
            say("Iteration {}/{}. EI: {}".format(i + 1, self.n_iter, ei), self.verbose, style="subtitle")
            self.optimizer.evaluate(setting, X, y)

        self.cv_results_, self.best_index_, self.best_estimator_ = self._create_cv_results()

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
        self._setup()
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
            "draw_samples": self.draw_samples
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

    def _setup(self):
        # Set maximum draw samples
        self.draw_samples = min(self.draw_samples, self.get_grid_size(self.encoded_params))

        # Limit the number of iterations for a grid that is too small
        self.n_iter = min(self.n_iter, self.get_grid_size(self.encoded_params))

        # Decode parameters for use inside the Model Optimizer
        self.decoded_params = decoder.decode_params(self.encoded_params)

        # Setup optimizer
        self.optimizer = Optimizer(estimator=self.estimator, param_distributions=self.decoded_params, inner_cv=10,
                                   scoring=self.scoring, timeout_score=self.timeout_score,
                                   max_eval_time=self.max_eval_time, use_ei_per_second=self.use_ei_per_second,
                                   verbose=self.verbose, draw_samples=self.draw_samples)

    def _create_cv_results(self):
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

        # Insert "params" and "mean_test_score" keywords
        cv_results = {
            "params": self.optimizer.validated_params,
            "mean_test_score": self.optimizer.validated_scores,
        }

        # Insert "param_*" keywords
        for setting in cv_results["params"]:
            for key, item in setting.items():
                param = "param_{}".format(key)
                if param not in cv_results:
                    cv_results[param] = []
                cv_results[param].append(item)

        # Find best index
        best_index = np.argmax(self.optimizer.validated_scores)
        best_setting = self.optimizer.validated_params[best_index]
        best_estimator = Builder.build_pipeline(self.estimator, best_setting)

        return cv_results, best_index, best_estimator
