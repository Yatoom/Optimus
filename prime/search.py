from optimus.search import Search
import numpy as np


class Optimizer:

    def __init__(self, models, scoring="accuracy", cv=10, verbose=True):
        self.optimi = []
        self.names = []
        self.cv = cv
        self.global_best_score = -np.inf
        self.global_best_time = np.inf
        self.verbose = verbose

        for model in models:
            self.optimi.append(Search(model["estimator"], model["params"], n_iter=None, population_size=100,
                                      scoring=scoring, cv=cv, n_jobs=-1, verbose=True))
            self.names.append(model["name"])

    def prepare(self, X, y, n_rounds=1, max_eval_time=150, max_retries=3):
        for index, optimus in enumerate(self.optimi):

            self._say("\nPreparing %s Optimizer with %s rounds" % (self.names[index], n_rounds))

            for iteration in range(0, n_rounds):

                self._say("---\nIteration %s/%s" % (iteration + 1, n_rounds))

                # Retry a few times to find a parameter that can be evaluated within max_eval_time.
                success = False
                for i in range(0, max_retries):
                    parameters, score = optimus.maximize(self.global_best_score, self.global_best_time)
                    success = optimus.evaluate(parameters, X, y, max_eval_time)

                    if success:
                        break

                # Drop if we never got success
                if not success:
                    self.optimi.pop(index)

                # Update global best score
                self.global_best_time = min(self.global_best_time, optimus.current_best_time)
                self.global_best_score = max(self.global_best_score, optimus.current_best_score)
            print("Best time:", self.global_best_time)

    def get_model_with_highest_ei(self):
        best_parameters = None
        best_optimus = None
        best_score = 0
        for optimus in self.optimi:
            parameters, score = optimus.maximize(self.global_best_score, self.global_best_time)

            if score > best_score:
                best_parameters = parameters
                best_optimus = optimus
                best_score = score

        return best_optimus, best_parameters

    def get_best_model(self):
        best_optimus = None
        best_score = 0
        for optimus in self.optimi:
            score = optimus.current_best_score
            if score > best_score:
                best_score = score
                best_optimus = optimus
        return best_optimus.get_best()

    def optimize(self, X, y, n_rounds, max_eval_time=150):
        for i in range(0, n_rounds):
            optimus, params = self.get_model_with_highest_ei()

            #
            if optimus is None:
                raise Exception("You need to call prepare() first.")

            index = self.optimi.index(optimus)
            self._say("---\nRound %s of %s. Running %s Optimizer" % (i+1, n_rounds, self.names[index]))

            optimus.evaluate(params, X, y, max_eval_time=max_eval_time, current_best_score=self.global_best_score)
            self.global_best_score = max(self.global_best_score, optimus.current_best_score)
        return self.get_best_model()

    def _say(self, *args):
        if self.verbose:
            print(*args)