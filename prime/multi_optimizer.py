from optimus.model_optimizer import ModelOptimizer
import numpy as np


class MultiOptimizer:

    def __init__(self, models, scoring="accuracy", cv=10, verbose=True, use_ei_per_second=False):
        """
        Optimizer for multiple models.
        :param models: Dictionary with a 'name': string, an 'estimator': Sklearn Estimator and 'params': dictionary.
        :param scoring: A Sklearn scorer: http://scikit-learn.org/stable/modules/model_evaluation.html
        :param cv: A Sklearn Cross-Validation object or number
        :param verbose: A boolean to control verbosity
        :param use_ei_per_second: Whether to use the standard EI or the EI / sqrt(second)
        """
        self.optimi = []
        self.names = []
        self.cv = cv
        self.global_best_score = -np.inf
        self.global_best_time = np.inf
        self.verbose = verbose
        self.results = []

        for model in models:
            self.optimi.append(ModelOptimizer(model["estimator"], model["params"], n_iter=None, population_size=100,
                                              scoring=scoring, cv=cv, verbose=True, use_ei_per_second=use_ei_per_second))
            self.names.append(model["name"])

    def prepare(self, X, y, n_rounds=1, max_eval_time=150, max_retries=3):
        """
        Prepare models by optimizing each model optimizer individually.
        :param X: List of features
        :param y: list of labels
        :param n_rounds: Number of rounds to initiate each model. Use string "auto" to automatically determine number 
        of rounds based on the number of parameters used. 
        :param max_eval_time: Maximum wall clock time in seconds for evaluating a single parameter setting
        :param max_retries: Maximum number of retries to find a parameter setting that does not result in an error. If
        the maximum number is exceeded, the model will be dropped. 
        """

        self._say("\nPreparing all models for %s rounds.\n----------------------------------" % n_rounds)

        # Keep a list of indices of model optimizers that could not successfully evaluate their parameters, so that we
        # can remove them later.
        to_remove = []

        for index, optimus in enumerate(self.optimi):

            degrees = n_rounds if n_rounds != "auto" else len(optimus.param_distributions)
            self._say("\nPreparing %s Optimizer with %s rounds" % (self.names[index], degrees))

            for iteration in range(0, degrees):

                self._say("---\nIteration %s/%s" % (iteration + 1, degrees))

                # Retry a few times to find a parameter that can be evaluated within max_eval_time.
                success = False
                for i in range(0, max_retries):
                    parameters, score = optimus.maximize(self.global_best_score, self.global_best_time)
                    success = optimus.evaluate(parameters, X, y, max_eval_time)

                    if success:
                        break

                # Drop if we never got success
                if not success:
                    to_remove.append(index)
                    break

                # Update global best score
                self.global_best_time = min(self.global_best_time, optimus.current_best_time)
                self.global_best_score = max(self.global_best_score, optimus.current_best_score)

                self.results.append({"score": optimus.current_best_score, "model": type(optimus.estimator).__name__})

        self.optimi = [i for j, i in enumerate(self.optimi) if j not in to_remove]
        self.names = [i for j, i in enumerate(self.names) if j not in to_remove]

    def get_model_with_highest_ei(self):
        """
        Ask each individual model optimizer to calculate its expected improvement and get the model with the best EI and
        the corresponding parameters. 
        :return: (best model optimizer, best parameters)
        """
        best_parameters = None
        best_optimus = None
        best_score = -np.inf
        for optimus in self.optimi:
            parameters, score = optimus.maximize(self.global_best_score, self.global_best_time)

            if score > best_score:
                best_parameters = parameters
                best_optimus = optimus
                best_score = score

        return best_optimus, best_parameters, best_score

    def get_best_model(self):
        """
        Gets the best model we have found so far. 
        :return: Returns the best model (Sklearn model), the name of the model (string) and its best score (float)
        """
        best_optimus = None
        best_score = -np.inf
        best_index = None
        for index, optimus in enumerate(self.optimi):
            score = optimus.current_best_score
            if score > best_score:
                best_score = score
                best_optimus = optimus
                best_index = index

        self._say("Best model: %s. Best score: %s" % (self.names[best_index], best_score))

        return best_optimus.get_best(), self.names[best_index], best_score

    def optimize(self, X, y, n_rounds, max_eval_time=150):
        """
        Finds the model and setting with the highest EI, and evaluates this model, for each round.
        :param X: List of features
        :param y: List of labels
        :param n_rounds: Number of rounds
        :param max_eval_time: Maximum wall clock time in seconds for evaluating a single parameter setting
        :return: If a best model could be found successfully, it will return the best model (Sklearn model), the name 
        of the model (string) and its best score (float). If all models failed, it will return None.
        """
        self._say("\nOptimizing for %s rounds.\n-------------------------\n" % n_rounds)

        for i in range(0, n_rounds):
            optimus, params, ei = self.get_model_with_highest_ei()

            if optimus is None:
                print("Either all models failed, or you need to call prepare() first.")
                return None

            index = self.optimi.index(optimus)
            self._say("---\nRound %s of %s. Running %s Optimizer with EI %s" % (i+1, n_rounds, self.names[index], ei))

            optimus.evaluate(params, X, y, max_eval_time=max_eval_time, current_best_score=self.global_best_score)
            self.global_best_score = max(self.global_best_score, optimus.current_best_score)
            self.results.append({"score": optimus.current_best_score, "model": type(optimus.estimator).__name__})
        return self.get_best_model()

    def reset(self):
        self.global_best_score = -np.inf
        self.global_best_time = np.inf
        for optimus in self.optimi:
            optimus.validated_params = []
            optimus.validated_scores = []
            optimus.validated_times = []
            optimus.current_best_setting = None
            optimus.current_best_score = -np.inf
            optimus.current_best_time = np.inf

    def get_all_validated_params(self):
        """
        Get all the validated parameters from the underlying model optimizers.
        :return: A dictionary where the name of the estimator maps to a dictionary where 'parameters' maps to its 
        validated parameters, and 'scores' maps to the corresponding scores. 
        """
        result = {}
        for optimus in self.optimi:
            # Get an official name and all validated parameters
            name = type(optimus.estimator).__name__
            params = optimus.validated_params
            scores = optimus.validated_scores

            # Check if name is unique, or add a number to make it unique
            i = 0
            new_name = name
            while new_name in result:
                new_name = "%s_%s" % (name, i)
                i += 1

            # Add parameters and scores to the results
            result[name] = {
                "parameters": params,
                "scores": scores,
                "best": float(np.max(scores))
            }

        return result

    def _say(self, *args):
        """
        Calls print() if verbose=True.
        :param args: Arguments to pass to print()
        """
        if self.verbose:
            print(*args)
