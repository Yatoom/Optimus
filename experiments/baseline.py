import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, \
    GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm
from xgboost import XGBRegressor

from optimus_ml.extra.forests import ExtraTreesRegressor

# Settings
target = "time"
file = "time-data.csv"

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

regressors = [
    LGBMRegressor(verbose=-1, min_child_samples=1),
    CatBoostRegressor(verbose=False, learning_rate=0.1, n_estimators=100),
    RandomForestRegressor(n_jobs=-1, n_estimators=100),
    ExtraTreesRegressor(n_jobs=-1, n_estimators=100),
    DecisionTreeRegressor(),
    GradientBoostingRegressor(),
    XGBRegressor(),
    KNeighborsRegressor(),
    LinearRegression(),
    LinearSVR(),
    AdaBoostRegressor(),
    # MLPRegressor()
]

for regressor in regressors:
    name = type(regressor).__name__
    durations = []
    losses = []

    for _ in tqdm(range(10)):
        for group in unique_groups:
            indices = np.where(np.array(groups) == group)[0]
            x = X[indices]
            y = Y_LOG[indices]

            kf = KFold(n_splits=10)
            kf.get_n_splits(x)

            for train_index, test_index in kf.split(x):
                X_train, X_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]

                start = time.time()
                regressor.fit(X_train, y_train)
                y_pred = regressor.predict(X_test)
                duration = time.time() - start
                y_pred = np.maximum(0, y_pred)
                loss = np.sqrt(np.mean((y_pred - y_test) ** 2))

                if np.isnan(loss):
                    print(np.isnan(loss))

                # Store results
                durations.append(duration)
                losses.append(loss)

    print(name, np.round(np.mean(losses), 4), np.round(np.mean(durations), 4))
print()
