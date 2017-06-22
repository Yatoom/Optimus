import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from benchmarks import config

db, table = config.connect()
seed = None


def visualize(filter_, x, y, statistic="mean", x_label=None, y_label=None, title="Results"):
    """
    Plot a graph of the benchmark data.

    Parameters
    ----------
    title: str
        Title for plot

    y_label: str
        Label for y-axis

    x_label: str
        Label for x-axis

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
    iteration_averages = calc_iteration_averages(df, statistic=statistic)

    # Visualize
    fig, ax = plt.subplots(figsize=(8, 6))
    x_label = x if x_label is None else x_label
    y_label = y if y_label is None else y_label
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    for label, df in iteration_averages.groupby("method"):
        df["cumulative_evaluation_time"] = np.cumsum(df["evaluation_time"])
        df.plot(x, y, ax=ax, label=label)

    plt.show()


def visualize_average(filter_=None, statistic="mean", x="cumulative_time", y="score", x_label=None, y_label=None,
                      title="Results"):
    if filter_ is None:
        filter_ = {}

    df = pd.DataFrame(list(table.find(filter_)))
    iteration_averages = calc_iteration_averages(df, statistic=statistic)

    fig, ax = plt.subplots(figsize=(8, 6))
    x_label = x if x_label is None else x_label
    y_label = y if y_label is None else y_label
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    colors = ["#16a085", "#C6F0DA", "#3498db", "#9b59b6", "#FF9696", "#f1c40f", "#e67e22", "#e74c3c", "#95a5a6",
              "#8BCBDE", "#45362E", "#63393E", "#B0DACC", "#3C3741", "#025159", "#EF9688", "#02135C"]
    pointer = 0
    for label, df in iteration_averages.groupby("method"):
        df["cumulative_evaluation_time"] = np.cumsum(df["evaluation_time"])
        df["avg_scores"] = averaging(df[y].tolist())
        df.plot(x=x, y="avg_scores", ax=ax, label=label, color=colors[pointer])
        pointer += 1

    plt.show()


def calc_iteration_averages(df, statistic="mean"):
    if statistic == "median":
        iteration_averages = df.groupby(["method", "iteration"]).median().reset_index()
    elif statistic == "max":
        iteration_averages = df.groupby(["method", "iteration"]).max().reset_index()
    else:
        iteration_averages = df.groupby(["method", "iteration"]).mean().reset_index()

    iteration_averages["best_rank"] = rank(iteration_averages, key="best_score")
    iteration_averages["rank"] = rank(iteration_averages, key="score")

    return iteration_averages


def rank(iteration_averages, key="best_score"):
    ranks = iteration_averages[["iteration", "method", key]].groupby(["iteration"]).rank(numeric_only=True, axis=0)

    return np.array(ranks[key]).tolist()


def averaging(numbers):
    averaged_numbers = []
    total = 0
    for index, number in enumerate(numbers):
        total += number
        averaged = total / (index + 1)
        averaged_numbers.append(averaged)
    return averaged_numbers


def visualize_benchmark(time_step, timestamp_key="cumulative_time", score_key="best_score", ranked=False,
                        averaged=False):
    data = make_time_buckets(time_step, timestamp_key, score_key)
    print("Plotting...")
    df = pd.DataFrame(data)

    if ranked:
        df = df.rank(axis=1)

    if averaged:
        df = df.apply(lambda x: np.cumsum(x) / np.arange(1, len(x) + 1), axis=0)

    df.plot()
    plt.show()


def make_time_buckets(time_step, timestamp_key, score_key):
    print("Fetching data...")
    df = pd.DataFrame(list(table.find({})))
    max_time = int(list(table.find({}).sort(timestamp_key, -1).limit(1))[0][timestamp_key])

    print("Reorganizing data...")
    methods = split(df, split_by="method")
    method_averages = get_method_averages(methods, time_step, max_time, timestamp_key, score_key)

    return method_averages


def split(df, split_by):
    methods = table.distinct(split_by)
    frames = []
    for method in methods:
        frames.append(df[df[split_by] == method])
    return frames


def get_method_averages(methods, time_step, max_time, timestamp_key, score_key):
    averages = {}
    for method in methods:
        tasks = split(method, split_by="task")
        average = get_task_average(tasks, time_step, max_time, timestamp_key, score_key)
        averages[str(method["method"].iloc[0])] = average

    return averages


def get_task_average(tasks, time_step, max_time, timestamp_key, score_key):
    scores_per_task = []
    for task in tasks:
        seeds = split(task, split_by="seed")
        seed_average = get_seed_average(seeds, time_step, max_time, timestamp_key, score_key)
        scores_per_task.append(seed_average)

    return np.average(scores_per_task, axis=0)


def get_seed_average(seeds, time_step, max_time, timestamp_key, score_key):
    scores_per_seed = []
    for sd in seeds:
        score_per_step = get_score_per_step(sd, time_step, max_time, timestamp_key, score_key)
        scores_per_seed.append(score_per_step)

    return np.average(scores_per_seed, axis=0)


def get_score_per_step(df, time_step, max_time, timestamp="cumulative_time", score="best_score"):
    times = np.array(df[timestamp])
    scores = np.array(df[score])
    pointer = 0
    score_per_step = []
    best_score = 0
    for time_point in np.arange(0, max_time + time_step, time_step):
        while pointer < len(times) and times[pointer] <= time_point:
            best_score = scores[pointer]
            pointer += 1
        score_per_step.append(best_score)

    return score_per_step
