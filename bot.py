from vault import models, data
import openml
import time

from prime.search import Optimizer

for t in [23, 36, 18, 31, 11, 53, 3647, 37, 49, 15, 21, 29, 37, 59]:
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
    openml_splits = data.get_openml_splits(task)
    prime = Optimizer(model_data, "accuracy", cv=openml_splits)
    prime.prepare(X, y, 1)
    best_model = prime.optimize(X, y, 50)
    print("Running OpenML task")
    run = openml.runs.run_task(task, best_model)
    run.publish()
    end = time.time() - start
    print("Published! Total time used: %s" % end)