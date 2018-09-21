import openml
import numpy as np
from optimus_ml import ModelOptimizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

task = openml.tasks.get_task(145677)
dataset = task.get_dataset()
X, y, categorical, names = dataset.get_data(
    target=dataset.default_target_attribute,
    return_categorical_indicator=True,
    return_attribute_names=True
)

oml_splits = [(j[0].train, j[0].test) for i, j in task.download_split().split[0].items()]
# openml_splits = [(i.train, i.test) for i in task.iterate_all_splits()]

# Setup classifier
clf = RandomForestClassifier(n_jobs=-1, random_state=3)

# Setup parameter grid
param_grid = {
    'n_estimators': [8, 16, 32],
    'criterion': ["gini", "entropy"],
    'max_features': np.arange(0.05, 0.5, 0.05).tolist(),
    'max_depth': [7, 8, 9, 10, 11, 12, None],
    'min_samples_split': list(range(2, 21)),
    'min_samples_leaf': list(range(1, 21)),
    'bootstrap': [True, False]
}

# Setup Model Optimizer
opt = ModelOptimizer(estimator=clf, encoded_params=param_grid, inner_cv=oml_splits, max_run_time=3000, n_iter=10,
                     local_search=True, scoring="roc_auc", score_regression="SMAC-forest")

# Fitting...
opt.fit(X, y)

# Print best parameter setting and corresponding score
a = pd.DataFrame(opt.cv_results_["params"])
a["score"] = opt.cv_results_["mean_test_score"]
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(a)