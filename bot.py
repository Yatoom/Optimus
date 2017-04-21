from vault import models as models
from prime.multi_optimizer import MultiOptimizer
import openml
import time


def get_openml_splits(openml_task):
    """
    Gets the splits.txt that are used in OpenML's task.
    :param openml_task: OpenMLTask object  
    :return: A list of splits.txt that can be used directly as a Sklearn CV
    """
    return [(i.train, i.test) for i in openml_task.iterate_all_splits()]

for t in [36, 18, 31, 11, 53, 3647, 37, 29, 15, 21, 37, 59, 23]:
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
    openml_splits = get_openml_splits(task)

    prime = MultiOptimizer(model_data, "accuracy", cv=openml_splits)
    prime.prepare(X, y, n_rounds="auto", max_eval_time=200, max_retries=4)
    best_model = prime.optimize(X, y, 75, max_eval_time=200)

    print("Running OpenML task")
    run = openml.runs.run_task(task, best_model)
    run.publish()
    end = time.time() - start
    print("Published! Total time used: %s" % end)

