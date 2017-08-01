from enum import Enum

import openml

from benchmarks import config
from optimus_ml.transcoder import converter
from optimus_ml.optimizer.model_optimizer import ModelOptimizer
from optimus_ml.vault import model_factory


class Method(Enum):
    RANDOMIZED = 0
    NORMAL = 1
    EI_PER_SECOND = 2
    EI_PER_ROOT_SECOND = 3
    RANDOMIZED_2X = 4


class Benchmark:
    def __init__(self, task_id, n_iter=200):
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()
        X, y, categorical, names = dataset.get_data(
            target=dataset.default_target_attribute,
            return_categorical_indicator=True,
            return_attribute_names=True
        )

        model = model_factory.generate_config(X, categorical, random_state=10)[0]
        estimator = converter.reconstruct_value(model["estimator"])
        params = model["params"]
        openml_splits = [(i.train, i.test) for i in task.iterate_all_splits()]

        self.X = X
        self.y = y
        self.estimator = estimator
        self.params = params
        self.openml_splits = openml_splits
        self.n_iter = n_iter
        self.task_id = task_id
        db, table = config.connect()
        self.db_table = table

    def benchmark(self, method=Method.RANDOMIZED, score_regressor="gp", time_regressor="gp", seed=0, starting_points=5,
                  verbose=False, local_search=False, classic=False):
        n_iter = self.n_iter
        if method == Method.RANDOMIZED:
            optimizer = ModelOptimizer(estimator=self.estimator, encoded_params=self.params, inner_cv=3, n_iter=n_iter,
                                       random_search=True, verbose=verbose)
        elif method == Method.NORMAL:
            optimizer = ModelOptimizer(estimator=self.estimator, encoded_params=self.params, inner_cv=3, n_iter=n_iter,
                                       random_search=False, verbose=verbose, use_ei_per_second=False,
                                       score_regression=score_regressor, local_search=local_search, classic=classic)
        elif method == Method.EI_PER_SECOND:
            optimizer = ModelOptimizer(estimator=self.estimator, encoded_params=self.params, inner_cv=3, n_iter=n_iter,
                                       random_search=False, verbose=verbose, use_ei_per_second=True,
                                       use_root_second=False, score_regression=score_regressor,
                                       time_regression=time_regressor, local_search=local_search, classic=classic)
        elif method == Method.EI_PER_ROOT_SECOND:
            optimizer = ModelOptimizer(estimator=self.estimator, encoded_params=self.params, inner_cv=3, n_iter=n_iter,
                                       random_search=False, verbose=verbose, use_ei_per_second=True,
                                       use_root_second=True, score_regression=score_regressor,
                                       time_regression=time_regressor, local_search=local_search, classic=classic)
        elif method == Method.RANDOMIZED_2X:
            optimizer = ModelOptimizer(estimator=self.estimator, encoded_params=self.params, inner_cv=3, n_iter=n_iter,
                                       random_search=True, verbose=verbose, simulate_speedup=2)

        optimizer.inner_cv = self.openml_splits

        # Let the first few points be always the same
        optimizer._setup()
        optimizer._random_search(self.X, self.y, starting_points, seed=seed)
        optimizer.fit(self.X, self.y, skip_setup=True)

        results = optimizer.cv_results_

        # Store results
        for i in range(0, len(results["best_score"])):
            iteration = {
                "task": self.task_id,
                # "method": "{}{}{} (EI: {}, RT: {})".format(
                #     method.name,
                #     "_LS_FIXED" if local_search else "",
                #     "_CL" if classic else "",
                #     score_regressor,
                #     time_regressor
                # ),
                "method": "{}{}{} (EI: {}, RT: {})".format(
                    method.name, "_LLS", "", score_regressor, time_regressor),
                "iteration": i,
                "score": results["mean_test_score"][i],
                "best_score": results["best_score"][i],
                "evaluation_time": results["evaluation_time"][i],
                "maximize_time": results["maximize_time"][i],
                "cumulative_time": results["cumulative_time"][i],
                "model": type(self.estimator).__name__,
                "params": results["readable_params"][i],
                "seed": seed,
                "method_name": method.name,
                "time_regressor": time_regressor,
                "score_regressor": score_regressor,
                "local_search": local_search,
                "classic": classic
            }
            self.db_table.insert(iteration)
