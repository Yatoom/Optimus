import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV

# Settings
target = "score"
file = "score-data.csv"
out = "score_250_samples_erf3.csv"
task = 12
use_log = True
scoring = "neg_mean_squared_error"
n_samples = 250
parameters = {
    "num_leaves": [4, 8, 16],
    "max_depth": [-1],
    "boosting_type": ["gbdt", "dart"],
    "learning_rate": [0.05, 0.1, 0.15],
    "n_estimators": [10, 25, 50, 100],
    "reg_alpha": [0, 0.10, 0.15, 0.20, 0.25]
}

# Load data
frame = pd.read_csv(file, index_col=0)

# Drop duplicates and impute max depth
frame = frame.drop_duplicates()
frame["max_depth_None"] = frame["max_depth"] == 'NoneType'
frame["max_depth"] = frame["max_depth"].replace("NoneType", 20).astype(int)

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
estimator = LGBMRegressor(verbose=-1, min_child_samples=1, objective="mse")
clf = GridSearchCV(estimator=estimator, param_grid=parameters, cv=10, verbose=10, scoring=scoring)

if use_log:
    clf.fit(X, Y_LOG)
else:
    clf.fit(X, Y)

# Gather results
results = pd.DataFrame(clf.cv_results_["params"])
results["fit_time"] = clf.cv_results_["mean_fit_time"]
results["target"] = clf.cv_results_["mean_test_score"]
results = results.sort_values("target")

results.to_csv(out)
