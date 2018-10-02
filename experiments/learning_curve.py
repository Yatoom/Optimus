import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Settings
from lightgbm import LGBMRegressor
from sklearn.model_selection import learning_curve

target = "time"
file = "time-data.csv"
task = 12
scoring = "neg_mean_squared_error"

# Full model
# params = {'boosting_type': 'gbdt', 'learning_rate': 0.1, 'max_depth': -1, 'n_estimators': 100, 'num_leaves': 128}

# Simple model
params = {'boosting_type': 'gbdt', 'learning_rate': 0.2, 'max_depth': -1, 'n_estimators': 100, 'num_leaves': 4}

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

est = LGBMRegressor(min_child_samples=1, **params)

train_sizes, train_scores, test_scores = learning_curve(est, X=X, y=Y_LOG, cv=10,
                                                        train_sizes=np.arange(50, 2000, 50),
                                                        scoring="neg_mean_squared_log_error", shuffle=True)

# Plot
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.grid()

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