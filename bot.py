from vault import models, data
import openml

from prime.search import Optimizer

for t in [23, 36, 18, 31, 11, 53, 3647, 37, 49, 15, 21, 29, 37, 59]:
    task = openml.tasks.get_task(t)

    dataset = task.get_dataset()
    X, y, categorical, names = dataset.get_data(
        target=dataset.default_target_attribute,
        return_categorical_indicator=True,
        return_attribute_names=True
    )

    models = models.get_models(categorical)
    openml_splits = data.get_openml_splits(task)
    prime = Optimizer(models, "accuracy", cv=openml_splits)
    prime.prepare(X, y, 1)
    best_model = prime.optimize(X, y, 50)
    run = openml.runs.run_task(task, best_model)
    run.publish()
