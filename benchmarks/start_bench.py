import benchmarks.benchmark as b

benchmark = b.Benchmark(49)
# benchmark.benchmark(method=b.Method.NORMAL, seed=500)
# benchmark.multi_bench(seeds=[33696, 97271,  9386, 13431, 55932, 63354, 51498, 51216, 33792, 25953])
benchmark.multi_bench(seeds=[33696, 97271,  9386])
