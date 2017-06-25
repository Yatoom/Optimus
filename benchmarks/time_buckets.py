import numpy as np
import pandas as pd

from benchmarks import config

db, table = config.connect()
default_seeds = [2589731706, 2382469894, 3544753667]


def plot(scores, ranked=False, averaged=False):
    frame = pd.DataFrame(scores)

    if ranked:
        frame = frame.rank(axis=1, ascending=False)

    if averaged:
        frame = frame.apply(lambda x: np.cumsum(x) / np.arange(1, len(x) + 1), axis=0)

    frame.plot()


def get_method_average(methods=None, tasks=None, seeds=None, step=1, max_time=None, time_key="cumulative_time",
                       score_key="best_score"):
    if methods is None:
        methods = table.distinct("method")

    if max_time is None:
        max_time = list(table.find({}).sort(time_key, -1).limit(1))[0][time_key]

    method_dict = {}
    for method in methods:
        task_average = get_task_average(method=method, tasks=tasks, seeds=seeds, step=step, max_time=max_time,
                                        time_key=time_key, score_key=score_key)
        method_dict[method] = task_average

    return method_dict


def get_task_average(method, tasks=None, seeds=None, step=1, max_time=1600, time_key="cumulative_time", score_key="best_score"):
    if tasks is None:
        tasks = table.distinct("task", {"method": method})

    scores_per_task = []
    for task in tasks:
        seed_average = get_seed_average(method=method, task=str(task), seeds=seeds, step=step, max_time=max_time,
                                        time_key=time_key, score_key=score_key)
        scores_per_task.append(seed_average)

    return np.average(scores_per_task, axis=0)


def get_seed_average(method, task, seeds=None, step=1, max_time=1600, time_key="cumulative_time",
                     score_key="best_score"):
    if seeds is None:
        seeds = table.distinct("seed", {"task": task})
        if len(seeds) == 0:
            print("No seeds found for method {}, task {}".format(method, task))

    scores_per_seed = []
    for seed in seeds:
        score_per_step = get_score_per_step(method=method, task=task, seed=seed, step=step, max_time=max_time,
                                            time_key=time_key, score_key=score_key)
        scores_per_seed.append(score_per_step)

    return np.average(scores_per_seed, axis=0)


def get_score_per_step(method, task, seed, step=1, max_time=1600, time_key="cumulative_time", score_key="best_score"):
    cursor = table.find({"method": method, "task": str(task), "seed": seed}).sort("iteration", 1)
    data = list(cursor)
    times = np.array([i[time_key] for i in data])
    scores = np.array([i[score_key] for i in data])

    pointer = 0
    score_per_step = []
    best_score = 0
    for time_point in np.arange(0, int(max_time) + 2 * step, step):
        while pointer < len(times) and times[pointer] <= time_point:
            best_score = scores[pointer]
            pointer += 1
        score_per_step.append(best_score)

    return score_per_step
