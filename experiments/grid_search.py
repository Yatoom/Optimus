import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV

# Settings
target = "score"
file = "score-data.csv"
out = "score_250_samples_erf2.csv"
task = 16
use_log = False
scoring = "neg_mean_squared_error"
n_samples = 250
parameters = {
    "max_leaf_nodes": [4, 16, 64, 128],
    "max_depth": [3, 5, 10, None],
    "max_features": ["sqrt", "log2", None],
    # "boosting_type": ["gbdt", "dart"],
    # "learning_rate": [0.1, 0.2, 0.3, 0.4, 0.5],
    "n_estimators": [10, 25, 50, 100]
}

# Load data
frame = pd.read_csv(file, index_col=0)

# Convert data
frame = pd.get_dummies(frame)
X = frame.drop(["group", target], axis=1)
X = np.array(X).astype(float)
Y = np.array(frame[target])
Y_LOG = np.log(Y + 1)
groups = np.array(frame["group"])
unique_groups = np.unique(groups)

# Take samples
np.random.seed(42)
indices = np.where(groups == task)[0]
if n_samples is not None:
    indices = np.random.choice(indices, n_samples)
X = X[indices]
Y = Y[indices]
Y_LOG = Y_LOG[indices]

# Run Grid Search
estimator = ExtraTreesRegressor(criterion="mse")
clf = GridSearchCV(estimator=estimator, param_grid=parameters, cv=10, verbose=10, scoring=scoring)

if use_log:
    clf.fit(X, Y_LOG)
else:
    clf.fit(X, Y)

# Gather results
frame = pd.DataFrame(clf.cv_results_["params"])
frame["fit_time"] = clf.cv_results_["mean_fit_time"]
frame["target"] = clf.cv_results_["mean_test_score"]
frame = frame.sort_values("target")

frame.to_csv(out)
