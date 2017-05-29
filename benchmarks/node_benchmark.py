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
table = db.Benchmark


def visualize(filter_):
    df = pd.DataFrame(list(table.find(filter_)))
    fold_averages = df.groupby(["method", "iteration"]).mean().drop("fold", 1).reset_index()

    # Visualize
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, df in fold_averages.groupby("method"):
        df["cumulative_evaluation_time"] = np.cumsum(df["evaluation_time"])
        df.plot(x="iteration", y="best_score", ax=ax, label=label)


def benchmark(task_id):
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
    benchmark_random(estimator, params, openml_splits, X, y, task_id)
    benchmark_normal(estimator, params, openml_splits, X, y, task_id)
    benchmark_root_second(estimator, params, openml_splits, X, y, task_id)


def benchmark_random(estimator, params, openml_splits, X, y, task_id):
    optimizer = ModelOptimizer(estimator=estimator, encoded_params=params, inner_cv=3, n_iter=50, random_search=True,
                               verbose=False)
    method = "random"

    for fold, (train, test) in enumerate(openml_splits):
        optimizer.fit(X[train], y[train])
        results = optimizer.cv_results_
        results["maximize_time"] = [0 for _ in optimizer.cv_results_["evaluation_time"]]
        store_fold(task_id, method, fold, results)


def benchmark_normal(estimator, params, openml_splits, X, y, task_id):
    optimizer = ModelOptimizer(estimator=estimator, encoded_params=params, inner_cv=3, n_iter=50, random_search=False,
                               verbose=False)
    method = "normal"

    for fold, (train, test) in enumerate(openml_splits):
        optimizer.fit(X[train], y[train])
        results = optimizer.cv_results_
        store_fold(task_id, method, fold, results)


def benchmark_root_second(estimator, params, openml_splits, X, y, task_id):
    optimizer = ModelOptimizer(estimator=estimator, encoded_params=params, inner_cv=3, n_iter=50, random_search=False,
                               verbose=False, use_ei_per_second=True)
    method = "root_second"

    for fold, (train, test) in enumerate(openml_splits):
        optimizer.fit(X[train], y[train])
        results = optimizer.cv_results_
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
