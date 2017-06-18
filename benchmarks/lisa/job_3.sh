#PBS -lwalltime=2:00:00
#PBS -lnodes=1
#PBS -lnodes=1:cpu3

# EI/s - GP/GP
~/python/python36/bin/python3 ~/Optimus/benchmark.py 145677 2 300 "gp" "gp" 5