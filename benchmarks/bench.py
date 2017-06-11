# This benchmark needs to be done one a single node.
import openml
from optimus.model_optimizer import ModelOptimizer
from vault import model_factory, decoder
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from benchmarks import config

db, table = config.connect()
seed = None


def visualize(filter_, x, y, statistic="mean"):
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

    statistic: "mean", "max" or "median"
        Show the mean, max or median of the iterations per fold
    """
    df = pd.DataFrame(list(table.find(filter_)))
    fold_averages = calc_fold_averages(df, statistic=statistic)

    # Visualize
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, df in fold_averages.groupby("method"):
        df["cumulative_evaluation_time"] = np.cumsum(df["evaluation_time"])
        df.plot(x, y, ax=ax, label=label)

    plt.show()


def visualize_average(filter_=None, statistic="mean", x="cumulative_time", y="score"):
    if filter_ is None:
        filter_ = {}

    df = pd.DataFrame(list(table.find(filter_)))
    fold_averages = calc_fold_averages(df, statistic=statistic)

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, df in fold_averages.groupby("method"):
        df["cumulative_evaluation_time"] = np.cumsum(df["evaluation_time"])
        df["avg_scores"] = averaging(df[y].tolist())
        df.plot(x=x, y="avg_scores", ax=ax, label=label)

    plt.show()


def calc_fold_averages(df, statistic="mean"):
    if statistic == "median":
        fold_averages = df.groupby(["method", "iteration"]).median().drop("fold", 1).reset_index()
    elif statistic == "max":
        fold_averages = df.groupby(["method", "iteration"]).max().drop("fold", 1).reset_index()
    else:
        fold_averages = df.groupby(["method", "iteration"]).mean().drop("fold", 1).reset_index()

    fold_averages["best_rank"] = rank(fold_averages, key="best_score")
    fold_averages["rank"] = rank(fold_averages, key="score")

    return fold_averages


def rank(fold_averages, key="best_score"):
    ranks = fold_averages[["iteration", "method", key]].groupby(["iteration"]).rank(numeric_only=True, axis=0)

    return np.array(ranks[key]).tolist()


def averaging(numbers):
    averaged_numbers = []
    total = 0
    for index, number in enumerate(numbers):
        total += number
        averaged = total / (index + 1)
        averaged_numbers.append(averaged)
    return averaged_numbers


def start_bench():
    global seed
    for s in [1517302035, 2148718844, 2739748925, 2713100086, 3803360706, 554072026, 975766179, 2463626752, 936287012,
              3326676407]:
        seed = s
        print("Seed: {}".format(seed))
        benchmark(49)


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
    print("Benchmarking random method")
    benchmark_random(estimator, params, openml_splits, X, y, task_id, simple=simple)

    print("\n Benchmarking EI method")
    benchmark_normal(estimator, params, openml_splits, X, y, task_id, simple=simple)

    print("\n Benchmarking EI per second method - GP")
    benchmark_ei_second(estimator, params, openml_splits, X, y, task_id, simple=simple, regression="gp")

    print("\n Benchmarking EI per root second method - GP")
    benchmark_root_second(estimator, params, openml_splits, X, y, task_id, simple=simple, regression="gp")

    print("\n Benchmarking EI per second method - linear")
    benchmark_ei_second(estimator, params, openml_splits, X, y, task_id, simple=simple, regression="linear")

    print("\n Benchmarking EI per root second method - linear")
    benchmark_root_second(estimator, params, openml_splits, X, y, task_id, simple=simple, regression="linear")


def benchmark_random(estimator, params, openml_splits, X, y, task_id, simple=True):
    optimizer = ModelOptimizer(estimator=estimator, encoded_params=params, inner_cv=3, n_iter=50, random_search=True,
                               verbose=False)
    fit_and_store(optimizer, openml_splits, X, y, task_id, simple, "Randomized", estimator, "None")


def benchmark_normal(estimator, params, openml_splits, X, y, task_id, simple=True):
    optimizer = ModelOptimizer(estimator=estimator, encoded_params=params, inner_cv=3, n_iter=50, random_search=False,
                               verbose=False)
    fit_and_store(optimizer, openml_splits, X, y, task_id, simple, "Normal", estimator, "None")


def benchmark_ei_second(estimator, params, openml_splits, X, y, task_id, simple=True, regression="gp"):
    optimizer = ModelOptimizer(estimator=estimator, encoded_params=params, inner_cv=3, n_iter=50, random_search=False,
                               verbose=False, use_ei_per_second=True, use_root_second=False, time_regression=regression)
    fit_and_store(optimizer, openml_splits, X, y, task_id, simple, "EI/s {}".format(regression), estimator, regression)


def benchmark_root_second(estimator, params, openml_splits, X, y, task_id, simple=True, regression="gp"):
    optimizer = ModelOptimizer(estimator=estimator, encoded_params=params, inner_cv=3, n_iter=50, random_search=False,
                               verbose=False, use_ei_per_second=True, use_root_second=True, time_regression=regression)
    fit_and_store(optimizer, openml_splits, X, y, task_id, simple, "EI/rs {}".format(regression), estimator, regression)


def fit_and_store(optimizer, openml_splits, X, y, task_id, simple, method, estimator, regression):
    if simple:
        optimizer.inner_cv = openml_splits

        # Let the first three points be always the same
        optimizer._setup()
        optimizer._random_search(X, y, 3, seed=seed)
        optimizer.fit(X, y, skip_setup=True)

        results = optimizer.cv_results_
        store_fold(task_id, method, 0, results, estimator, regression)
    else:
        for fold, (train, test) in enumerate(openml_splits):
            optimizer.fit(X[train], y[train])
            results = optimizer.cv_results_
            results["maximize_time"] = [0 for _ in optimizer.cv_results_["evaluation_time"]]
            store_fold(task_id, method, fold, results, estimator)


def store_fold(task_id, method, fold, results, estimator, time_regressor):
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
            "model": type(estimator).__name__,
            "params": results["readable_params"][i],
            "seed": seed,
            "time_regressor": time_regressor
        }
        table.insert(iteration)
