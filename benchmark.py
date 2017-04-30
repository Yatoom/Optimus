from vault import models as models
from prime.multi_optimizer import MultiOptimizer
import openml
import time
import sys
import numpy as np

print(sys.argv)
logfile = sys.argv[1]
n_jobs = int(sys.argv[2])
job_number = int(sys.argv[3])

print(logfile, n_jobs, job_number)

OPENML_100 = [12, 14, 16, 20, 22, 28, 32, 41, 45, 58, 14967, 14969, 14970, 125920, 3543, 3510, 3512, 3567, 14968, 34538,
              125921, 125922, 9950, 9954, 9955, 9956, 9960, 9964, 9968, 9970, 9976, 9977, 3481, 3561, 9979, 9981, 9985,
              9986, 14964, 3904, 2075, 2079, 3022, 3021, 2074, 3948, 3946, 7592, 3560, 3549, 24, 43, 23, 34539, 3913,
              9967, 34537, 3954, 3896, 3485, 125923, 3492, 3, 9978, 9980, 3902, 3918, 34536, 9914, 9946, 9971, 219,
              3889, 2, 14966, 10101, 14965, 6, 3493, 3903, 49, 21, 3950, 31, 11, 15, 18, 9952, 9957, 36, 37, 29, 53,
              10093, 14971, 3891, 3917, 3494, 9983, 3899]


for t in np.split(np.array(OPENML_100), n_jobs)[job_number]:
    print("\n===\nStarting task %s\n===\n" % t)
    start = time.time()
    task = openml.tasks.get_task(t)

    dataset = task.get_dataset()
    X, y, categorical, names = dataset.get_data(
        target=dataset.default_target_attribute,
        return_categorical_indicator=True,
        return_attribute_names=True
    )

    model_data = models.get_models(categorical, X, random_state=3)
    openml_splits = [(i.train, i.test) for i in task.iterate_all_splits()]

    prime = MultiOptimizer(model_data, "accuracy", cv=openml_splits, use_ei_per_second=False)
    prime.prepare(X, y, n_rounds="auto", max_eval_time=300, max_retries=4)
    best_model = prime.optimize(X, y, 50, max_eval_time=300)

    # Store
    duration = round(time.time() - start, 1)

    # Default text to store in log
    text = "%s, %s, %s, %s\n" % (t, "error", 0, duration)

    # If successful, override text with the model name and score
    if best_model is not None:
        model, name, score = best_model
        text = "%s, %s, %s, %s\n" % (t, name, score, duration)
        try:
            print("Running OpenML task")
            run = openml.runs.run_task(task, model)
            run.publish()
        except:
            pass

    with open(logfile, "a+") as file:
        file.write(text)