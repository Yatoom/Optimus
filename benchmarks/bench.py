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
        fold_averages = df.groupby(["method", "iteration"]).median().reset_index()
    elif statistic == "max":
        fold_averages = df.groupby(["method", "iteration"]).max().reset_index()
    else:
        fold_averages = df.groupby(["method", "iteration"]).mean().reset_index()

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
