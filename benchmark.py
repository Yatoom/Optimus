import sys

import benchmarks.benchmark as b

# Example:
# benchmark.py [task_id] [method] [seed] [score_regressor] [time_regressor] [starting_points] [local search]
# [use_projection]
# benchmark.py 145677 2 300 "gp" "gp" 5 0 1

task_id = sys.argv[1]
method = b.Method(int(sys.argv[2]))
seed = int(sys.argv[3])
score_regressor = sys.argv[4]
time_regressor = sys.argv[5]
starting_points = int(sys.argv[6])
local_search = bool(int(sys.argv[7]))

print(task_id, method, seed, score_regressor, time_regressor, starting_points, local_search)

bench = b.Benchmark(task_id, n_iter=20000)
bench.benchmark(method=method, seed=seed, score_regressor=score_regressor, time_regressor=time_regressor,
                starting_points=starting_points, verbose=False, local_search=local_search)
