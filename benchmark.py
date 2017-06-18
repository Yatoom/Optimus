import benchmarks.benchmark as b
import sys

# Example:
# benchmark.py 145677 2 300 "gp" "gp" 5

task_id = sys.argv[1]
method = b.Method(int(sys.argv[2]))
seed = int(sys.argv[3])
score_regressor = sys.argv[4]
time_regressor = sys.argv[5]
starting_points = int(sys.argv[6])


bench = b.Benchmark(task_id, n_iter=20000)
bench.benchmark(method=method, seed=seed, score_regressor=score_regressor, time_regressor=time_regressor,
                starting_points=starting_points, verbose=False)
