import openml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from extra.dual_imputer import DualImputer
from vault.models import make_conditional_steps
from wrapper.optimus import OptimusCV
from wrapper.prime import PrimeCV
import numpy as np

task = openml.tasks.get_task(49)

dataset = task.get_dataset()
X, y, categorical, names = dataset.get_data(
    target=dataset.default_target_attribute,
    return_categorical_indicator=True,
    return_attribute_names=True
)

any_missing = bool(np.isnan(X).any())
DI = DualImputer(categorical=categorical)
OHE = OneHotEncoder(categorical_features=categorical, handle_unknown="ignore", sparse=False)
SS = StandardScaler()
# model = PrimeCV(scoring="accuracy", cv=3, verbose=True, use_ei_per_second=False, prep_rounds=1,
#                 opt_rounds=5, max_eval_time=15, max_prep_retries=2, categorical=categorical, missing=any_missing)

model = OptimusCV(LogisticRegression(n_jobs=-1, penalty='l2', random_state=3), {
    'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
    'dual': [True, False],
    "@preprocessor": make_conditional_steps([
        (DI, any_missing, None),
        ([DI, SS], any_missing, SS)
    ])
})

run = openml.runs.run_task(task, model)
run.publish()
