# This benchmark needs to be done one a single node.
import openml
from optimus.model_optimizer import ModelOptimizer
from vault import model_factory, decoder
from pymongo import MongoClient
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

client = MongoClient('localhost', 27017)
db = client.local
table = db.NewBenchmark


def visualize(filter_, x, y):
    """
    Plot a graph of the benchmark data.

    Parameters
    ----------
    filter_: dict
        A MongoDB filter, for example {"task": 49}
    x: str
        What data to use on the x-axis. For example "iteration".
    y: str
        What data to use on the y-axis. For example "best_score".

    """
    df = pd.DataFrame(list(table.find(filter_)))
    fold_averages = df.groupby(["method", "iteration"]).mean().drop("fold", 1).reset_index()

    # Visualize
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, df in fold_averages.groupby("method"):
        df["cumulative_evaluation_time"] = np.cumsum(df["evaluation_time"])
        df.plot(x, y, ax=ax, label=label)


def benchmark(task_id, simple=True):
    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    X, y, categorical, names = dataset.get_data(
        target=dataset.default_target_attribute,
        return_categorical_indicator=True,
        return_attribute_names=True
    )

    # Logistic regression
    model = model_factory.generate_config(X, categorical, random_state=10)[0]
    estimator = decoder.decode_source_tuples(model["estimator"], special_prefix="!")
    params = model["params"]
    openml_splits = [(i.train, i.test) for i in task.iterate_all_splits()]

    # Benchmark individual
    # print("Benchmarking random method")
    # benchmark_random(estimator, params, openml_splits, X, y, task_id, simple=simple)

    print("\n Benchmarking normal method")
    benchmark_normal(estimator, params, openml_splits, X, y, task_id,  simple=simple)

    print("\n Benchmarking root ei method")
    benchmark_root_second(estimator, params, openml_splits, X, y, task_id, simple=simple)


def benchmark_random(estimator, params, openml_splits, X, y, task_id, simple=True):
    optimizer = ModelOptimizer(estimator=estimator, encoded_params=params, inner_cv=3, n_iter=50, random_search=True,
                               verbose=False)
    fit_and_store(optimizer, openml_splits, X, y, task_id, simple, "random")


def benchmark_normal(estimator, params, openml_splits, X, y, task_id, simple=True):
    optimizer = ModelOptimizer(estimator=estimator, encoded_params=params, inner_cv=3, n_iter=50, random_search=False,
                               verbose=False)
    fit_and_store(optimizer, openml_splits, X, y, task_id, simple, "normal")


def benchmark_root_second(estimator, params, openml_splits, X, y, task_id, simple=True):
    optimizer = ModelOptimizer(estimator=estimator, encoded_params=params, inner_cv=3, n_iter=50, random_search=False,
                               verbose=False, use_ei_per_second=True)
    fit_and_store(optimizer, openml_splits, X, y, task_id, simple, "root_second")


def fit_and_store(optimizer, openml_splits, X, y, task_id, simple, method):
    if simple:
        optimizer.inner_cv = openml_splits
        optimizer.fit(X, y)
        results = optimizer.cv_results_
        store_fold(task_id, method, 0, results)
    else:
        for fold, (train, test) in enumerate(openml_splits):
            optimizer.fit(X[train], y[train])
            results = optimizer.cv_results_
            results["maximize_time"] = [0 for _ in optimizer.cv_results_["evaluation_time"]]
            store_fold(task_id, method, fold, results)


def store_fold(task_id, method, fold, results):
    for i in range(0, len(results["best_score"])):
        iteration = {
            "task": task_id,
            "method": method,
            "fold": fold,
            "iteration": i,
            "score": results["mean_test_score"][i],
            "best_score": results["best_score"][i],
            "evaluation_time": results["evaluation_time"][i],
            "maximize_time": results["maximize_time"][i],
            "cumulative_time": results["cumulative_time"][i],

        }
        table.insert(iteration)
