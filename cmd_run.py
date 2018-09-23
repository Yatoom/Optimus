import sys
import json
import os
from benchmarking.benchmarker import Benchmarker

arguments = sys.argv
task = int(arguments[1])
seed = int(arguments[2])
config_index = int(arguments[3])

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
filename = os.path.join(dname, "benchmarking/jobs.json")
with open(filename, "r+") as f:
    job = json.load(f)[config_index]

b = Benchmarker(task, 10000)
b.benchmark(**job)