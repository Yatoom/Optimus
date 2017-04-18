from vault import models as models, data
from prime.multi_optimizer import MultiOptimizer
import openml
import time


def get_openml_splits(task):
    """
    Gets the splits that are used in OpenML's task.
    :param task: 
    :return: 
    """
    generator = task.iterate_all_splits()
    splits = [(i.train, i.test) for i in generator]
    return splits


# 36, 18, 31, 11, 53, 3647, 37, 49,
for t in [15, 21, 29, 37, 59, 23]:
    print("\n===\nStarting task %s\n===\n" % t)
    start = time.time()
    task = openml.tasks.get_task(t)

    dataset = task.get_dataset()
    X, y, categorical, names = dataset.get_data(
        target=dataset.default_target_attribute,
        return_categorical_indicator=True,
        return_attribute_names=True
    )

    model_data = models.get_models(categorical, X)
    openml_splits = get_openml_splits(task)

    prime = MultiOptimizer(model_data, "accuracy", cv=openml_splits)
    prime.prepare(X, y, 3, max_eval_time=150)
    best_model = prime.optimize(X, y, 50)
    print("Running OpenML task")
    run = openml.runs.run_task(task, best_model)
    run.publish()
    end = time.time() - start
    print("Published! Total time used: %s" % end)
