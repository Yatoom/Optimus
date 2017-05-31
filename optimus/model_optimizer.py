from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection._search import BaseSearchCV, ParameterSampler
from optimus.optimizer import Optimizer
from extra.fancyprint import say
from vault import decoder
from tqdm import tqdm


class ModelOptimizer(BaseSearchCV):
    def __init__(self, estimator, encoded_params, inner_cv: object = None, scoring="accuracy", timeout_score=0,
                 max_eval_time=120, use_ei_per_second=False, verbose=True, draw_samples=100, n_iter=10, refit=True,
                 random_search=False):
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
            
        max_eval_time: int or float
            Time in seconds until evaluation times out
            
        refit: boolean, default=True
            Refit the best estimator with the entire dataset. If "False", it is impossible to make predictions using
            the estimator instance after fitting
            
        random_search: boolean
            If true, use random search instead of bayesian search 
        """

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
        self.random_search = random_search

        # Placeholders for derived variables
        self.draw_samples = draw_samples
        self.n_iter = n_iter
        self.best_estimator_ = None
        self.best_index_ = None
        self.cv_results_ = None
        self.decoded_params = None
        self.optimizer = None

    def fit(self, X, y):
        """
        Optimize the model for `n_rounds`.

        Parameters
        ----------
        X: array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples.

        y: array of shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels) as integers or strings.

        """

        # Calculate derived variables
        self._setup()

        # Use Randomized Search or Bayesian Optimization
        if self.random_search:
            self._random_search(X, y)
        else:
            self._bayesian_search(X, y)

        # Store results
        self.cv_results_, self.best_index_, self.best_estimator_ = self.optimizer.create_cv_results()

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

    def _bayesian_search(self, X, y):
        say("Bayesian search with {} iterations".format(self.n_iter), self.verbose, style="title")

        for i in tqdm(range(0, self.n_iter), ascii=False, leave=True):
            setting, ei = self.optimizer.maximize(realize=False)
            say("Iteration {}/{}. EI: {}".format(i + 1, self.n_iter, ei), self.verbose, style="subtitle")
            self.optimizer.evaluate(setting, X, y)

    def _random_search(self, X, y):
        say("Randomized search with {} iterations".format(self.n_iter), self.verbose, style="title")
        samples = [i for i in ParameterSampler(self.decoded_params, self.n_iter)]

        for i in tqdm(range(0, self.n_iter), ascii=False, leave=True):
            setting = samples[i]
            say("Iteration {}/{}.".format(i + 1, self.n_iter), self.verbose, style="subtitle")
            self.optimizer.evaluate(setting, X, y)

    def _setup(self):

        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        # Set maximum draw samples, and limit the number of iterations for a grid that is too small
        grid_size = self.get_grid_size(self.encoded_params)
        self.draw_samples = min(self.draw_samples, grid_size)
        self.n_iter = min(self.n_iter, grid_size)

        say("Maximum number of iterations as limited by grid: {}".format(grid_size), verbose=self.verbose)

        # Decode parameters for use inside the Model Optimizer
        self.decoded_params = decoder.decode_params(self.encoded_params)

        # Setup optimizer
        self.optimizer = Optimizer(estimator=self.estimator, param_distributions=self.decoded_params,
                                   inner_cv=self.inner_cv, scoring=self.scoring, timeout_score=self.timeout_score,
                                   max_eval_time=self.max_eval_time, use_ei_per_second=self.use_ei_per_second,
                                   verbose=self.verbose, draw_samples=self.draw_samples)
