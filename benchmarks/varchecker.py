import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from benchmarks import config

db, table = config.connect()
default_seeds = [2589731706, 2382469894, 3544753667]
default_tasks = [12, 14, 16, 20, 22, 28, 32, 45, 58]


def get_data(method="NORMAL_LS (-0.05) (EI: gp, RT: gp)", task=12, seed=default_seeds[0]):
    cursor = table.find({"method": method, "task": str(task), "seed": seed}).sort("iteration", 1)
    return list(cursor)


def get_var(data):
    start_time = time.time()
    scores = [i["score"] for i in data]
    return np.var(scores), len(scores), time.time() - start_time