from sklearn.model_selection import cross_val_score, ParameterSampler
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from optimus.converter import Converter
from scipy.stats import norm, warnings
from optimus.builder import Builder
from extra.timeout import Timeout
from extra.fancyprint import say
from sklearn import clone
import numpy as np
import traceback
import time

warnings.filterwarnings("ignore")


class Optimizer:
    def __init__(self, estimator, param_distributions, inner_cv=10, scoring="accuracy", timeout_score=0,
                 max_eval_time=120, use_ei_per_second=False, verbose=True, draw_samples=100):
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
            out different preprocessors. The values of parameters that start with a "!" will be source decoded, and 
            stored under a new key without the prefix.  
        
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
            Whether to use the standard EI or the EI / sqrt(second)
            
        verbose: bool
            Whether to print extra information
            
        draw_samples: int
            Number of randomly selected samples we maximize over 
        """

        # Accept parameters
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.inner_cv = inner_cv
        self.scoring = scoring
        self.timeout_score = timeout_score
        self.max_eval_time = max_eval_time
        self.use_ei_per_second = use_ei_per_second
        self.verbose = verbose
        self.draw_samples = draw_samples

        # Setup initial values
        self.validated_scores = []
        self.validated_params = []
        self.validated_times = []
        self.converted_params = None
        self.current_best_score = -np.inf
        self.current_best_time = np.inf

        # Setting up the Gaussian Process Regressors
        cov_amplitude = ConstantKernel(1.0, (0.01, 1000.0))
        other_kernel = Matern(
            length_scale=np.ones(len(self.param_distributions)),
            length_scale_bounds=[(0.01, 100)] * len(self.param_distributions),
            nu=2.5)

        gp = GaussianProcessRegressor(
            kernel=cov_amplitude * other_kernel,
            normalize_y=True, random_state=3, alpha=0.0,
            n_restarts_optimizer=2)

        self.gp_score = gp  # type: GaussianProcessRegressor
        self.gp_time = clone(gp)  # type: GaussianProcessRegressor

    def __str__(self):
        # Returns the name of the estimator (e.g. LogisticRegression)
        return type(self.estimator).__name__

    def maximize(self):
        """
        Find the best setting - the setting with the highest expected improvement - to evaluate.
        
        Returns
        -------
        best_setting: dict
            The setting with the highest expected improvement
        
        best_score: float
            The highest EI (per second)
        """

        # Select a sample of parameters
        sampled_params = ParameterSampler(self.param_distributions, self.draw_samples)

        # Determine the best parameters
        return self._maximize_on_sample(sampled_params)

    def evaluate(self, parameters, X, y):
        """
        Evaluates a parameter setting and updates the list of validated parameters afterward.
        
        Parameters
        ----------
        parameters: list
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
                score = cross_val_score(best_estimator, X, y, scoring=self.scoring, cv=self.inner_cv, n_jobs=-1)

        except (GeneratorExit, OSError, TimeoutError):
            say("Timeout error :(")
            success = False
            score = [self.timeout_score]
        except RuntimeError:
            # It might be that it still works when we set n_jobs to 1 instead of -1.
            # Below we check if we can set n_jobs to 1 and if so, recall this function again.
            if "n_jobs" in self.estimator.get_params() and self.estimator.get_params()["n_jobs"] != 1:
                say("Runtime error, trying again with n_jobs=1.")
                self.estimator.set_params(n_jobs=1)
                return self.evaluate(parameters, X, y)

            # Otherwise, we're going to catch the error as we normally do
            else:
                say("RuntimeError")
                print(traceback.format_exc())
                success = False
                score = [self.timeout_score]

        except Exception:
            say("An error occurred with parameters", Converter.readable_parameters(parameters))
            print(traceback.format_exc())
            success = False
            score = [self.timeout_score]

        running_time = time.time() - start if success else self.max_eval_time

        # Get the mean and store the results
        score = np.mean(score)  # type: float
        self.validated_scores.append(score)
        self.validated_params.append(parameters)
        self.converted_params = Converter.convert_settings(self.validated_params, self.param_distributions)
        self.validated_times.append(running_time)
        self.current_best_time = min(running_time, self.current_best_time)
        self.current_best_score = max(score, self.current_best_score)

        say("Score: %s | best: %s | time: %s" % (score, self.current_best_score, running_time))

        return success, score, running_time

    def _maximize_on_sample(self, sampled_params):
        """
        Finds the next best setting to evaluate from a set of samples. 
        
        Parameters
        ----------
        sampled_params: list
            The samples to calculate the expected improvement on

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

        # Check if the number of validated scores (without timeouts) is zero
        if num_valid_scores == 0:
            return np.random.choice([i for i in sampled_params]), 0

        # Fit parameters
        try:
            self.gp_score.fit(self.converted_params, self.validated_scores)
            if self.use_ei_per_second:
                self.gp_time.fit(self.converted_params, self.validated_times)
        except:
            print(traceback.format_exc())

        best_score = - np.inf
        best_setting = None
        for setting in sampled_params:
            converted_setting = Converter.convert_setting(setting, self.param_distributions)
            score = self._get_ei_per_second(converted_setting, self.current_best_score)
            # print("Score: %s| Setting: %s " % (score, Converter.readable_parameters(setting)))
            if score > best_score:
                best_score = score
                best_setting = setting

        return best_setting, self._realize(best_setting, best_score)

    def _realize(self, best_setting, original):
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

        Returns
        -------
        Returns the realistic estimate
        """

        params, scores = Converter.remove_timeouts(self.validated_params, self.validated_scores, self.timeout_score)

        if len(scores) == 0:
            return original

        converted_settings = Converter.convert_settings(params, self.param_distributions)
        self.gp_score.fit(converted_settings, scores)

        if self.use_ei_per_second:
            times, _ = Converter.remove_timeouts(self.validated_times, self.validated_scores, self.timeout_score)
            self.gp_time.fit(converted_settings, times)

        setting = Converter.convert_setting(best_setting, self.param_distributions)

        return self._get_ei_per_second(setting, self.current_best_score)

    def _get_ei_per_second(self, point, current_best_score):
        """
        Calculate the expected improvement and divide it by the square root of the estimated validation time.
        
        Parameters
        ----------
        point:
            Setting to predict on
            
        current_best_score:
            The current score optimum
            
        Returns
        -------
        Return the EI / sqrt(estimated seconds) or the EI, depending on the use_ei_per_second value
        """

        ei = self._get_ei(point, current_best_score)

        if self.use_ei_per_second:
            seconds = self.gp_time.predict(point)
            return ei / np.sqrt(seconds)
        return ei

    def _get_ei(self, point, current_best_score):
        """
        Calculate the expected improvement.
        
        Parameters
        ----------
        point: list
            Parameter setting for the GP to predict on
        current_best_score: float
            The current score optimum
        Returns
        -------
        Returns the Expected Improvement
        """

        # Extra check. This way seems to work around rounding errors, and it can be computed surprisingly fast.
        if point in self.gp_score.X_train_.tolist():
            return 0

        point = np.array(point).reshape(1, -1)
        mu, sigma = self.gp_score.predict(point, return_std=True)
        best_score = current_best_score
        mu = mu[0]
        sigma = sigma[0]

        # We want our mu to be higher than the best score
        # We subtract 0.01 because http://haikufactory.com/files/bayopt.pdf
        # (2.3.2 Exploration-exploitation trade-of)
        # Intuition: makes diff less important, while sigma becomes more important
        diff = mu - best_score  # - 0.01

        # If sigma is zero, this means we have already seen that point, so we do not need to evaluate it again.
        # We use a value slightly higher than 0 in case of small machine rounding errors.
        if sigma <= 1e-05:
            return 0

        # Expected improvement function
        Z = diff / sigma
        ei = diff * norm.cdf(Z) + sigma * norm.pdf(Z)

        return ei
