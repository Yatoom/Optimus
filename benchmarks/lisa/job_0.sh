#PBS -lwalltime=2:00:00
#PBS -lnodes=1
#PBS -lnodes=1:cpu3

# RANDOMIZED SEARCH
~/python/python36/bin/python3 ~/Optimus/benchmark.py 145677 0 300 "gp" "gp" 5
