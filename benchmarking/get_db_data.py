import numpy as np
import openml
import pandas as pd

import benchmarking.config as cfg
from optimus_ml.transcoder import converter
from optimus_ml.vault import model_factory


# Get the parameter distribution for a certain task
def get_param_distribution_for_task(task_id):
    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    X, y, categorical, names = dataset.get_data(
        target=dataset.default_target_attribute,
        return_categorical_indicator=True,
        return_attribute_names=True
    )

    model = model_factory.generate_config(X, categorical, random_state=10)[0]
    params = model["params"]

    encoded_params = params
    result = converter.reconstruct_grid(encoded_params)
    result = converter.dictionary_to_readable_dictionary(result)
    return result


# Convert parameters to integers
def convert_params(group, params, times):
    unique_groups = np.unique(group)
    models = {i: get_param_distribution_for_task(i) for i in unique_groups}

    new_groups = []
    new_settings = []
    new_times = []
    for g in unique_groups:
        indices = np.where(np.array(group) == g)[0]
        model = models[g]
        settings = converter.settings_to_indices(np.array(params)[indices], model)
        new_times += np.array(times)[indices].tolist()
        new_groups += np.array(group)[indices].tolist()
        new_settings += settings

    return new_groups, new_settings, new_times


db, table = cfg.connect()
target = "score"
group = [i["task"] for i in table.find({})]
params = [i["params"] for i in table.find({})]
time = [i[target] for i in table.find({})]

# g, p, t = convert_params(group, params, time)

frame = pd.DataFrame(params)
frame["group"] = group
frame[target] = time

frame.to_csv(f"{target}-data.csv")
