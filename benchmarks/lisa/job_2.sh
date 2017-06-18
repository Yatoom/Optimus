#PBS -lwalltime=2:00:00
#PBS -lnodes=1
#PBS -lnodes=1:cpu3

# NORMAL SEARCH - Forest
~/python/python36/bin/python3 ~/Optimus/benchmark.py 145677 1 300 "forest" "gp" 5