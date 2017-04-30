import openml
from wrapper.optimus import Optimus
import numpy as np

task = openml.tasks.get_task(49)

dataset = task.get_dataset()
X, y, categorical, names = dataset.get_data(
    target=dataset.default_target_attribute,
    return_categorical_indicator=True,
    return_attribute_names=True
)

any_missing = bool(np.isnan(X).any())
model = Optimus(scoring="accuracy", cv=3, verbose=True, use_ei_per_second=False, prep_rounds=1,
                opt_rounds=5, max_eval_time=15, max_prep_retries=2, categorical=categorical, missing=any_missing)

run = openml.runs.run_task(task, model)
run.publish()
