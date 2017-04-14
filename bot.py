from vault import models, data
import openml

from prime.search import Optimizer

task = openml.tasks.get_task(2071)

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
best_model = prime.optimize(X, y, 20)
run = openml.runs.run_task(task, best_model)
run.publish()
