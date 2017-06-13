import benchmarks.benchmark as b

benchmark = b.Benchmark(49)
# benchmark.benchmark(method=b.Method.NORMAL, seed=500)
benchmark.multi_bench(seeds=[500, 1000, 1500])
