import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from extra.dual_imputer import DualImputer
from vault.transformers import Choice, Discrete, LogScale


def get_models(categorical, features):
    # Pre-processing
    DI = DualImputer(categorical=categorical)
    OHE = OneHotEncoder(categorical_features=categorical, handle_unknown="ignore", sparse=False)
    SS = StandardScaler()
    RS = RobustScaler()
    PC = PCA()
    PF = PolynomialFeatures()

    # Are there missing values?
    any_missing = np.isnan(features).any()

    # Are there any categorical features?
    any_categorical = np.any(categorical)

    print("Dataset has missing values: %s. Dataset has categorical values: %s" % (any_missing, any_categorical))

    models = [
        {
            "name": "Random Forest",
            "estimator": RandomForestClassifier(n_jobs=-1, n_estimators=300, random_state=3),
            "params": {
                "criterion": Choice(["gini", "entropy"]),
                "max_depth": Choice([3, 5, 7, 9, None]),
                "max_features": Choice(np.arange(0.05, 0.5, 0.05)),
                "min_samples_leaf": Discrete((1, 21)),
                "min_samples_split": Discrete((2, 21)),
                'bootstrap': Choice([True, False]),
                '@preprocessor': Choice(make_conditional_steps([
                    (DI, any_missing, None),
                    (OHE, not any_missing and any_categorical),
                    ([DI, OHE], any_missing and any_categorical)
                ]))
            }
        },
        {
            "name": "SVM Kernels",
            "estimator": SVC(probability=True),
            "params": {
                "C": LogScale(2, (-10, 10)),
                "gamma": LogScale(2, (-10, 0)),
                "kernel": Choice(["linear", "poly", "rbf"]),
                "@preprocessor": Choice(make_conditional_steps([
                    (DI, any_missing, None),
                    ([DI, SS], any_missing, SS)
                ]))
            }
        },
        {
            "name": "Gradient Boosting",
            "estimator": GradientBoostingClassifier(random_state=3, n_estimators=512),
            "params": {
                "max_depth": Choice([1, 2, 3]),
                "learning_rate": LogScale(10, (-5, 0)),
                "@preprocessor": Choice(make_conditional_steps([
                    (DI, any_missing, None)
                ]))
            }
        },
        {
            "name": "K-Neighbors",
            "estimator": KNeighborsClassifier(n_jobs=-1),
            "params": {
                'n_neighbors': Choice([1, 3, 5, 7, 9]),
                'weights': Choice(["uniform", "distance"]),
                'p': Choice([1, 2]),
                "@preprocessor": Choice(make_conditional_steps([
                    (DI, any_missing, None),
                    ([DI, SS], any_missing, SS)
                ]))
            }
        },
        {
            "name": "LogisticRegression",
            "estimator": LogisticRegression(n_jobs=-1, penalty="l2", random_state=3),
            "params": {
                'C': Choice([1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.]),
                'dual': Choice([True, False]),
                "@preprocessor": Choice(make_conditional_steps([
                    (DI, any_missing, None),
                    ([DI, SS], any_missing, SS)
                ]))
            }
        },
    ]

    return models


def make_conditional_steps(conditional_steps):
    """
    Adds preprocessors to a list if a condition is fulfilled 
    :param conditional_steps: (Preprocessor steps, condition, [alternative]) 
    :return: the list of preprocessors that fulfill their conditions
    """
    result = []
    for conditional_step in conditional_steps:

        if len(conditional_step) == 3:
            step, condition, alternative = conditional_step

            if condition:
                result.append(step)
            else:
                result.append(alternative)

        else:
            step, condition = conditional_step
            if condition:
                result.append(step)

    return result
