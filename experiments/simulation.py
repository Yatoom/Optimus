import time

import numpy as np
import pandas as pd

# Settings
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
from smac.epm.rf_with_instances import RandomForestWithInstances

target = "score"
score_file = "score-data.csv"
time_file = "time-data.csv"

# Load data
score_frame = pd.read_csv(score_file, index_col=0)
time_frame = pd.read_csv(time_file, index_col=0)

# Drop duplicates and impute max depth
subset = ["@preprocessor", "bootstrap", "criterion", "max_depth", "min_samples_leaf", "min_samples_split", "n_estimators"]
score_frame = score_frame.drop_duplicates(subset=subset)
time_frame = time_frame.drop_duplicates(subset=subset)
score_frame["max_depth_None"] = score_frame["max_depth"] == 'NoneType'
score_frame["max_depth"] = score_frame["max_depth"].replace("NoneType", 20).astype(int)

# Drop zero-scores
valid = score_frame["score"] > 0
score_frame = score_frame[valid]
time_frame = time_frame[valid]

# Convert data
score_frame = pd.get_dummies(score_frame)
X = score_frame.drop(["group", target], axis=1)
X = np.array(X).astype(float)
Y = np.array(score_frame[target])
duration = time_frame["time"].tolist()
groups = np.array(score_frame["group"])
unique_groups = np.unique(groups)

# Wrapper
class Wrap(RandomForestWithInstances):
    def __init__(self, types, bounds, *args, **kwargs):
        super().__init__(types=types, bounds=bounds, *args, **kwargs)

    def fit(self, X, y):
        return self.train(X, y)

    def predict(self, X: np.ndarray):
        means, vars = super().predict(X)
        return means

    def predict_mean_variance(self, X: np.ndarray):
        return super().predict(X)

    def get_params(self, deep=False):
        return {
            "types": self.types,
            "bounds": self.bounds,
        }

    def set_params(self, deep=False, **kwargs):
        pass


# Setup LightGBM model
lgbm_time_estimator = LGBMRegressor(verbose=-1, min_child_samples=1, objective="mse", num_leaves=4, reg_alpha=0.20, learning_rate=0.15)
lgbm_score_estimator = LGBMRegressor(verbose=-1, min_child_samples=1, objective="quantile", num_leaves=8)
lgbm_mean_estimator = LGBMRegressor(verbose=-1, min_child_samples=1, objective="mse", num_leaves=8)

# Setup RFR
bounds_ = list(zip(X.min(axis=0), X.max(axis=0)))
types_ = [0 for _ in range(len(bounds_))]
rfr_score_estimator = Wrap(np.array(types_), np.array(bounds_))
rfr_time_estimator = Wrap(np.array(types_), np.array(bounds_))
# lgbm_time_estimator = rfr_time_estimator
# rfr_time_estimator = lgbm_time_estimator

# Set mode
# lgbm_mode = True
# use_eips = True

for mode in ["gbqr", "rfr", "rfr-rfr", "rfr-gbm"]:
# for mode in ["rfr-gbm"]:


    # Predict EI
    def predict_ei(points):
        if mode == "gbqr":
            ei = lgbm_score_estimator.predict(points)
            ei[observed] = -1
            return ei
        elif mode in ["rfr", "rfr-rfr", "rfr-gbm"]:
            mu, var = rfr_score_estimator.predict_mean_variance(points)
            mu = mu.reshape(-1)
            var = var.reshape(-1)
            sigma = np.sqrt(var)
            diff = mu - np.max(observed_Y)
            Z = diff / sigma
            ei = diff * norm.cdf(Z) + sigma * norm.pdf(Z)

            if mode == "rfr-rfr":
                run_time = rfr_time_estimator.predict(points).reshape(-1)
                eips = ei / np.maximum(0, run_time)
                eips[observed] = -1
                return eips
            elif mode == "rfr-gbm":
                run_time = lgbm_time_estimator.predict(points).reshape(-1)
                eips = ei / np.maximum(0, run_time)
                eips[observed] = -1
                return eips

            ei[observed] = -1
            return ei
        raise RuntimeError(f"Mode {mode} unknown.")

            # # LightGBM
            # booster = estimator.booster_
            # estimates = []
            # for tree_id, leaf_ids in enumerate(result.T):
            #     mapping = {leaf_id: booster.get_leaf_output(tree_id, leaf_id) for leaf_id in np.unique(leaf_ids)}
            #     individual_estimates = np.zeros(len(leaf_ids))
            #     for id, value in mapping.items():
            #         individual_estimates[leaf_ids == id] = value
            #     estimates.append(individual_estimates)
            #
            # estimates = np.array(estimates).T
            # mu = np.sum(estimates, axis=1)
            # sigma = np.std(estimates, axis=1)


    results = {}
    fitting_times_per_task = {}
    prediction_times_per_task = {}
    evaluation_times_per_task = {}
    # for task_id in [12, 14, 16, 20, 22, 28, 32, 41, 45, 58]:
    for task_id in [12, 14, 16, 20, 22, 28, 32, 41, 45, 58]:
        # Select by task
        indices = np.where(groups == task_id)[0]
        X_task = X[indices]
        Y_task = Y[indices]
        duration_task = np.array(duration)[indices]

        # Start with 3 random data points
        observed = np.zeros(len(X_task)).astype(bool)
        observed[:3] = True
        observed_X = X_task[:3].tolist()
        observed_Y = Y_task[:3].tolist()
        observed_duration = duration_task[:3].tolist()

        fitting_times = [0,0,0]
        prediction_times = [0,0,0]

        for i in range(500):
            start = time.time()
            if mode == "gbqr":
                lgbm_score_estimator.fit(np.array(observed_X), np.array(observed_Y))
            else:
                rfr_score_estimator.fit(np.array(observed_X), np.array(observed_Y))

                if mode == "rfr-gbm":
                    lgbm_time_estimator.fit(np.array(observed_X), np.log(np.array(observed_duration) + 1))
                elif mode == "rfr-rfr":
                    rfr_time_estimator.fit(np.array(observed_X), np.log(np.array(observed_duration) + 1))
            fitting_times.append(time.time() - start)

            start = time.time()
            eis = predict_ei(X_task)
            prediction_times.append(time.time() - start)

            index = np.argmax(eis)
            observed[index] = True

            observed_X.append(X_task[index])
            observed_Y.append(Y_task[index])
            observed_duration.append(duration_task[index])
            print(i, index, X_task[index].tolist(), Y_task[index])

        results[task_id] = np.maximum.accumulate(np.array(observed_Y) / np.max(Y_task))
        fitting_times_per_task[task_id] = fitting_times
        prediction_times_per_task[task_id] = prediction_times
        evaluation_times_per_task[task_id] = observed_duration


    fit_frame = pd.DataFrame(fitting_times_per_task)
    pred_frame = pd.DataFrame(prediction_times_per_task)
    duration_frame = pd.DataFrame(evaluation_times_per_task)
    score_frame = pd.DataFrame(results)

    prefix = mode
    fit_frame.to_csv(f"sim/500-{prefix}-fit-time.csv")
    pred_frame.to_csv(f"sim/500-{prefix}-time.csv")
    duration_frame.to_csv(f"sim/500-{prefix}-eval-time.csv")
    score_frame.to_csv(f"sim/500-{prefix}-scores.csv")



        # frame = pd.DataFrame(results)
        # print()
        #
        # std = frame.std(axis=1)
        # mean = frame.mean(axis=1)
        #
        # plt.plot(mean, color="blue")
        # plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2, color="blue")
        # plt.show()


        #
        # # Score
        # X_train, X_test = X[observed], X[~observed]
        # Y_train, Y_test = Y[observed], Y[~observed]
        #
        # # upper.fit(X_train, Y_train)
        # upper.fit(X_train, Y_train)
        # Y_pred = upper.predict(X_test)
        # score = mean_squared_error(Y_test, Y_pred)
        #
        # print(score, f"{np.max(observed_Y)}/{np.max(Y)}", np.argmax(observed_Y))
        # print()
