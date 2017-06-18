import subprocess

for i in range(0, 7):
    subprocess.call(["qsub", "./job_{}.sh".format(i)])