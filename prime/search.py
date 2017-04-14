from optimus.search import Search


class Optimizer:

    def __init__(self, models, scoring="accuracy", cv=10, verbose=True):
        self.optimi = []
        self.names = []
        self.cv = cv
        self.global_best_score = 0
        self.verbose = verbose

        for model in models:
            self.optimi.append(Search(model["estimator"], model["params"], n_iter=None, population_size=100,
                                      scoring=scoring, cv=cv, n_jobs=-1, verbose=True))
            self.names.append(model["name"])

    def prepare(self, X, y, n_rounds=1):
        for index, optimus in enumerate(self.optimi):
            self._say("Preparing %s Optimizer with %s rounds" % (self.names[index], n_rounds))
            for iteration in range(0, n_rounds):
                self._say("---\nIteration %s/%s" % (iteration + 1, n_rounds))
                parameters, score = optimus.sample_and_maximize(self.global_best_score)
                optimus.evaluate(parameters, X, y)
            self.global_best_score = max(self.global_best_score, optimus.current_best_score)

    def get_model_with_highest_ei(self):
        best_parameters = None
        best_optimus = None
        best_score = 0
        for optimus in self.optimi:
            parameters, score = optimus.sample_and_maximize(self.global_best_score)

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

    def optimize(self, X, y, n_rounds):
        for i in range(0, n_rounds):
            optimus, params = self.get_model_with_highest_ei()
            if optimus is None:
                raise Exception("You need to call prepare() first.")
            optimus.evaluate(params, X, y)
        return self.get_best_model()

    def _say(self, *args):
        if self.verbose:
            print(*args)