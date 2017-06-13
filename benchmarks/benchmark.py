import openml

from optimus.model_optimizer import ModelOptimizer
from vault import model_factory, decoder
from benchmarks import config
from enum import Enum
import numpy as np


class Method(Enum):
    RANDOMIZED = 0
    NORMAL = 1
    EI_PER_SECOND = 2
    EI_PER_ROOT_SECOND = 3


class Benchmark:
    def __init__(self, task_id, inner_cv=3, n_iter=50):
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()
        X, y, categorical, names = dataset.get_data(
            target=dataset.default_target_attribute,
            return_categorical_indicator=True,
            return_attribute_names=True
        )

        model = model_factory.generate_config(X, categorical, random_state=10)[0]
        estimator = decoder.decode_source_tuples(model["estimator"], special_prefix="!")
        params = model["params"]
        openml_splits = [(i.train, i.test) for i in task.iterate_all_splits()]

        self.X = X
        self.y = y
        self.estimator = estimator
        self.params = params
        self.openml_splits = openml_splits
        self.inner_cv = inner_cv
        self.n_iter = n_iter
        self.task_id = task_id
        db, table = config.connect()
        self.db_table = table

    def multi_bench(self, seeds=None):
        if seeds is None:
            seeds = [1517302035, 2148718844, 2739748925, 2713100086, 3803360706, 554072026, 975766179, 2463626752,
                     936287012, 3326676407]

        for seed in seeds:
            print("Seed: {}".format(seed))

            print("Randomized")
            self.benchmark(method=Method.RANDOMIZED, seed=seed)

            print("Normal - GP")
            self.benchmark(method=Method.NORMAL, seed=seed, score_regressor="gp")

            print("Normal - Forest")
            self.benchmark(method=Method.NORMAL, seed=seed, score_regressor="forest")

            print("EI/s - GP/GP")
            self.benchmark(method=Method.EI_PER_SECOND, seed=seed, score_regressor="gp", time_regressor="gp")

            print("EI/s - GP/Linear")
            self.benchmark(method=Method.EI_PER_SECOND, seed=seed, score_regressor="gp", time_regressor="linear")

            print("EI/s - Forest/GP")
            self.benchmark(method=Method.EI_PER_SECOND, seed=seed, score_regressor="forest", time_regressor="gp")

            print("EI/s - Forest/Linear")
            self.benchmark(method=Method.EI_PER_SECOND, seed=seed, score_regressor="forest", time_regressor="linear")

            print("EI/√s - GP/GP")
            self.benchmark(method=Method.EI_PER_ROOT_SECOND, seed=seed, score_regressor="gp", time_regressor="gp")

            print("EI/√s - GP/Linear")
            self.benchmark(method=Method.EI_PER_ROOT_SECOND, seed=seed, score_regressor="gp", time_regressor="linear")

            print("EI/√s - Forest/GP")
            self.benchmark(method=Method.EI_PER_ROOT_SECOND, seed=seed, score_regressor="forest", time_regressor="gp")

            print("EI/√s - Forest/Linear")
            self.benchmark(method=Method.EI_PER_ROOT_SECOND, seed=seed, score_regressor="forest", time_regressor="linear")

    def benchmark(self, method=Method.RANDOMIZED, score_regressor="gp", time_regressor="gp", seed=0, starting_points=3):
        if method == Method.RANDOMIZED:
            optimizer = ModelOptimizer(estimator=self.estimator, encoded_params=self.params, inner_cv=3, n_iter=50,
                                       random_search=True, verbose=False)
        elif method == Method.NORMAL:
            optimizer = ModelOptimizer(estimator=self.estimator, encoded_params=self.params, inner_cv=3, n_iter=50,
                                       random_search=False, verbose=False, use_ei_per_second=False,
                                       score_regression=score_regressor)
        elif method == Method.EI_PER_SECOND:
            optimizer = ModelOptimizer(estimator=self.estimator, encoded_params=self.params, inner_cv=3, n_iter=50,
                                       random_search=False, verbose=False, use_ei_per_second=True,
                                       use_root_second=False, score_regression=score_regressor,
                                       time_regression=time_regressor)
        elif method == Method.EI_PER_ROOT_SECOND:
            optimizer = ModelOptimizer(estimator=self.estimator, encoded_params=self.params, inner_cv=3, n_iter=50,
                                       random_search=False, verbose=False, use_ei_per_second=True,
                                       use_root_second=True, score_regression=score_regressor,
                                       time_regression=time_regressor)

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
                "method": "{} (EI: {}, RT: {})".format(method.name, score_regressor, time_regressor),
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
                "score_regressor": score_regressor
            }
            self.db_table.insert(iteration)
