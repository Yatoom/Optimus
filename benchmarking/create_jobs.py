import json
import os

python_path = "/home/jhoof/python/python36/bin/python3"
project_dir = "/home/jhoof/Optimus"
config = "#PBS -lwalltime=1:00:00 -lnodes=1:cpu3"
tasks = [12, 14, 16, 20, 22, 28, 32, 41, 45, 58]
# seeds = [2589731706, 2382469894, 3544753667]
seeds = [2589731706]

with open("jobs.json", "r+") as f:
    jobs = json.load(f)
    job_indices = range(len(jobs))

if not os.path.exists("jobs"):
    os.mkdir("jobs")

counter = 0
for task in tasks:
    for seed in seeds:
        for index in job_indices:
            description = f"{config}\n{python_path} {project_dir}/cmd_run.py {task} {seed} {index}"
            with open(f"jobs/{task}_{seed}_{index}.sh", "w+") as f:
                f.write(description)
            print(f"Job for task {task} with seed {seed} and configuration {index} created.")
            counter += 1
print(f"{counter} jobs created.")
