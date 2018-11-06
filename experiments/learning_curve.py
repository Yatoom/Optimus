import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Settings
from lightgbm import LGBMRegressor
from sklearn.model_selection import learning_curve
from smac.epm.rf_with_instances import RandomForestWithInstances

target = "score"
file = "score-data.csv"
task = 41
scoring = "neg_mean_squared_error"

class Wrap(RandomForestWithInstances):
    def __init__(self, types, bounds, *args, **kwargs):
        super().__init__(types=types, bounds=bounds, *args, **kwargs)

    def fit(self, X, y):
        return self.train(X, y)

    def predict(self, X: np.ndarray):
        means, vars = super().predict(X)
        return means

    def get_params(self, deep=False):
        return {
            "types": self.types,
            "bounds": self.bounds,
        }

    def set_params(self, deep=False, **kwargs):
        pass

# Full model
# params = {'boosting_type': 'gbdt', 'learning_rate': 0.1, 'max_depth': -1, 'n_estimators': 100, 'num_leaves': 128}

# Simple model
# params = {'boosting_type': 'gbdt', 'learning_rate': 0.2, 'max_depth': -1, 'n_estimators': 100, 'num_leaves': 4}
# params = {'boosting_type': 'gbdt', 'learning_rate': 0.15, 'max_depth': -1, 'n_estimators': 100, 'num_leaves': 4, "reg_alpha": 0.15}
# params = {'boosting_type': 'gbdt', 'learning_rate': 0.15, 'max_depth': -1, 'n_estimators': 100, 'num_leaves': 4, "reg_alpha": 0.20}

# Simple model for scoring
params = {'boosting_type': 'gbdt', 'learning_rate': 0.15, 'max_depth': -1, 'n_estimators': 100, 'num_leaves': 4, "reg_alpha": 0}

# Load data
frame = pd.read_csv(file, index_col=0)

# Drop duplicates, get task, and impute max depth
frame = frame.drop_duplicates()
frame = frame[frame["score"] != 0]
frame = frame[frame["group"] == task]
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

est = LGBMRegressor(min_child_samples=1, objective="mse", **params)
# bounds_ = list(zip(X.min(axis=0), X.max(axis=0)))
# types_ = [0 for _ in range(len(bounds_))]
# est = Wrap(np.array(types_), np.array(bounds_))

np.random.seed(42)
train_sizes, train_scores, test_scores = learning_curve(est, X=X, y=Y_LOG, cv=10,
                                                        train_sizes=np.arange(50, 2000, 50),
                                                        scoring="neg_mean_squared_error", shuffle=True)

# Plot
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.grid()
# plt.ylim(-0.005, 0.001)

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
         label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
         label="Cross-validation score")

plt.legend(loc="best")
plt.show()