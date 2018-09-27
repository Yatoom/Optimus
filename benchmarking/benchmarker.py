import openml

from benchmarking import config
from optimus_ml import ModelOptimizer
from optimus_ml.transcoder import converter
from optimus_ml.vault import model_factory


class Benchmarker:
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
        openml_splits = [(j[0].train, j[0].test) for i, j in task.download_split().split[0].items()]

        self.X = X
        self.y = y
        self.estimator = estimator
        self.params = params
        self.openml_splits = openml_splits
        self.n_iter = n_iter
        self.task_id = task_id
        db, table = config.connect()
        self.db_table = table

    def benchmark(self, name, label, seed, starting_points, max_run_time=1500, store_results=True, **optimizer_args):
        optimizer = ModelOptimizer(estimator=self.estimator, encoded_params=self.params, n_iter=self.n_iter,
                                   inner_cv=3, max_run_time=max_run_time, **optimizer_args)

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
                "method": label,
                "iteration": i,
                "score": results["mean_test_score"][i],
                "best_score": results["best_score"][i],
                "evaluation_time": results["evaluation_time"][i],
                "maximize_time": results["maximize_time"][i],
                "cumulative_time": results["cumulative_time"][i],
                "model": type(self.estimator).__name__,
                "params": results["readable_params"][i],
                "seed": seed,
                "method_name": name,
                "time_regressor": optimizer.time_regression,
                "score_regressor": optimizer.score_regression,
                "local_search": optimizer.local_search,
                "classic": optimizer.multi_start
            }

            if store_results:
                self.db_table.insert(iteration)
        return self

if __name__ == "__main__":
    job = {
        "name": "GP EIPS",
        "label": "GP/LightGBM EIPS",
        "starting_points": 5,
        "use_ei_per_second": True,
        "score_regression": "gp",
        "time_regression": "lightgbm",
        "local_search": False,
        "xi": 0
    }

    b = Benchmarker(31, 10000)
    b.benchmark(seed=100, max_run_time=30, store_results=False, **job)
