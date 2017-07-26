import subprocess
import os

python_path = "/home/jhoof/python/python36/bin/python3"
project_dir = "/home/jhoof/Optimus/"
config = "#PBS -lwalltime=2:00:00\n#PBS -lnodes=1:cpu3\n"


def benchmark_1():
    OPENML_10 = [12, 14, 16, 20, 22, 28, 32, 41, 45, 58]
    SEEDS = [2589731706, 2382469894, 3544753667]

    for task in OPENML_10:
        for seed in SEEDS:
            create_all_jobs(task, seed)

    sub_jobs()


def create_all_jobs(task, seed):
    # Randomized 2X
    # create_job(task, 4, seed)

    # # Randomized
    # create_job(task, 0, seed)
    #
    # Normal - GP
    # create_job(task, 1, seed, local_search=local_search, score_regressor="gp")

    # Normal - GP + LS + Projection
    # create_job(task, 1, seed, local_search=True, score_regressor="gp", use_projection=True)

    # Normal - normal forest + LS
    # create_job(task, 1, seed, local_search=True, score_regressor="normal forest")

    create_job(task, 1, seed, score_regressor="forest")


    # Normal - Forest
    # create_job(task, 1, seed, local_search=local_search, score_regressor="forest")

    # EI/s - GP/GP
    # create_job(task, 2, seed, local_search=local_search, score_regressor="gp", time_regressor="gp")

    # EI/s - GP / Extra forest
    # create_job(task, 2, seed, local_search=local_search, score_regressor="gp", time_regressor="extra forest")

    # # EI/s - Forest / Forest
    # create_job(task, 2, seed, local_search=local_search, score_regressor="forest", time_regressor="forest")
    #
    # # EI/s - Forest / Extra forest
    # create_job(task, 2, seed, local_search=local_search, score_regressor="forest", time_regressor="extra forest")

    # EI/s - Forest - Linear
    # create_job(task, 2, seed, local_search=local_search, score_regressor="forest", time_regressor="linear")


def get_number_of_jobs():
    return len(os.listdir(project_dir + "benchmarks/lisa/jobs/"))


def create_job(task, method, seed, local_search=False, score_regressor="gp", time_regressor="gp", starting_points=5):
    n_jobs = get_number_of_jobs()
    f = open('jobs/job_{}.sh'.format(n_jobs), "w+")
    f.write(config)
    benchmark_path = project_dir + "benchmark.py"
    f.write("{0} {1} {2} {3} {4} \"{5}\" \"{6}\" {7} {8}".format(python_path, benchmark_path, task, method, seed,
                                                                     score_regressor, time_regressor, starting_points,
                                                                     int(local_search)))


def sub_jobs():
    for i in range(0, get_number_of_jobs()):
        subprocess.call(["qsub", "./jobs/job_{}.sh".format(i)])
