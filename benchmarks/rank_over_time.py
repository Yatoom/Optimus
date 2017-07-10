import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from benchmarks import config

db, table = config.connect()
default_seeds = [2589731706, 2382469894, 3544753667]


def multiplot():
    # Ranking
    scores = get_method_ranking(seeds=default_seeds, max_time=1500)
    plot(rename(scores), y_label="mean of method ranking over different tasks", x_label="time(s)",
         title="Comparison of different classifiers for calculating expected improvement and expected running time.")

    # Speed (iterations over time)
    scores = get_method_ranking(seeds=default_seeds, max_time=1500, score_key="iteration")
    plot(rename(scores), y_label="iterations", x_label="time(s)", title="Speed of different methods")

    # Evaluation time
    scores = get_method_ranking(seeds=default_seeds, max_time=1500, score_key="evaluation_time")
    plot(rename(scores), y_label="iterations", x_label="time(s)", title="Evaluation time of different methods")

    # Maximize time
    scores = get_method_ranking(seeds=default_seeds, max_time=1500, score_key="maximize_time")
    plot(rename(scores), y_label="iterations", x_label="time(s)", title="Maximization time of different methods")


def plot(scores, averaged=False, x_label="", y_label="", title=""):
    frame = pd.DataFrame(scores)

    if averaged:
        frame = frame.apply(lambda x: np.cumsum(x) / np.arange(1, len(x) + 1), axis=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    frame.plot(ax=ax)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.show()


def rename(scores):
    return {
        "Randomized": scores["RANDOMIZED (EI: gp, RT: gp)"],
        "Normal (gp)": scores["NORMAL (EI: gp, RT: gp)"],
        "Normal (forest)": scores["NORMAL (EI: forest, RT: gp)"],
        "EI/s (forest / extra forest)": scores["EI_PER_SECOND (EI: forest, RT: extra forest)"],
        "EI/s (forest / forest)": scores["EI_PER_SECOND (EI: forest, RT: forest)"],
        "EI/s (forest / linear)": scores["EI_PER_SECOND (EI: forest, RT: linear)"],
        "EI/s (gp / gp)": scores["EI_PER_SECOND (EI: gp, RT: gp)"]
    }


def get_method_average(methods=None, tasks=None, seeds=None, step=1, max_time=None, time_key="cumulative_time",
                       score_key="best_score"):
    method_task_averages = {}

    if methods is None:
        methods = table.distinct("method")

    methods.remove("RANDOMIZED_2X (EI: gp, RT: gp)")

    if max_time is None:
        max_time = list(table.find({}).sort(time_key, -1).limit(1))[0][time_key]

    for method in methods:
        scores_per_task = get_scores_per_task(method=method, tasks=tasks, seeds=seeds, step=step, max_time=max_time,
                                              time_key=time_key, score_key=score_key)

        if scores_per_task is not None:
            method_task_averages[method] = np.average(scores_per_task, axis=0)

    return method_task_averages


def get_method_ranking(methods=None, tasks=None, seeds=None, step=1, max_time=None, time_key="cumulative_time",
                       score_key="best_score"):
    method_task_scores = {}

    if methods is None:
        methods = table.distinct("method")

    methods.remove("RANDOMIZED_2X (EI: gp, RT: gp)")

    if max_time is None:
        max_time = list(table.find({}).sort(time_key, -1).limit(1))[0][time_key]

    for method in methods:
        scores_per_task = get_scores_per_task(method=method, tasks=tasks, seeds=seeds, step=step, max_time=max_time,
                                              time_key=time_key, score_key=score_key)

        if scores_per_task is not None:
            method_task_scores[method] = scores_per_task

    first_key = list(method_task_scores.keys())[0]
    n_tasks = len(method_task_scores[first_key])
    ranks = {method: [] for method in methods}

    for i in range(0, n_tasks):
        single_task_scores = {method: method_task_scores[method][i] for method in methods}
        ranked = pd.DataFrame(single_task_scores).rank(axis=1, ascending=False)
        ranked_dict = ranked.to_dict(orient="list")
        for key, values in ranked_dict.items():
            ranks[key].append(values)

    avg = {method: [] for method in methods}
    for method, tasks in ranks.items():
        avg[method] = np.average(tasks, axis=0)

    return avg


# The average over the tasks will form the method score
def get_scores_per_task(method, tasks=None, seeds=None, step=1, max_time=1600, time_key="cumulative_time",
                        score_key="best_score"):
    if tasks is None:
        tasks = table.distinct("task", {"method": method})

    scores_per_task = []
    for task in tasks:
        seed_average = get_seed_average(method=method, task=str(task), seeds=seeds, step=step, max_time=max_time,
                                        time_key=time_key, score_key=score_key)

        if seed_average is not None:
            scores_per_task.append(seed_average)

    if len(scores_per_task) == 0:
        return None

    return scores_per_task


# The average over the seeds will form the task score
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
        if score_per_step is not None:
            scores_per_seed.append(score_per_step)

    if len(scores_per_seed) == 0:
        return None

    return np.average(scores_per_seed, axis=0)


def get_score_per_step(method, task, seed, step=1, max_time=1600, time_key="cumulative_time", score_key="best_score"):
    cursor = table.find({"method": method, "task": str(task), "seed": seed}).sort("iteration", 1)
    data = list(cursor)
    times = np.array([i[time_key] for i in data])
    scores = np.array([i[score_key] for i in data])

    if len(times) == 0:
        return None

    print("method: {}, task: {}, seed: {}, data: {}".format(method, task, seed, len(times)))
    times, scores = remove_startup_iterations(times, scores, 5, recalculate_time=True)

    pointer = 0
    score_per_step = []
    best_score = 0
    for time_point in np.arange(0, int(max_time) + 2 * step, step):
        while pointer < len(times) and times[pointer] <= time_point:
            best_score = scores[pointer]
            pointer += 1
        score_per_step.append(best_score)

    return score_per_step


def remove_startup_iterations(times, scores, nr, recalculate_time=True):
    startup_time = times[nr - 1]
    new_times = times[nr:]
    new_scores = scores[nr:]

    if recalculate_time:
        for i in range(0, len(new_times)):
            new_times[i] = new_times[i] - startup_time

    return new_times, new_scores
