from copy import copy, deepcopy
from vault import decoder
import numpy as np


def generate_config(X, categorical, random_state):
    # Meta-features to calculate from X
    any_missing = bool(np.isnan(X).any())
    any_categorical = bool(np.any(categorical))

    print("Any missing:", any_missing)
    print("Any categorical", any_categorical)

    # Pre-processing operators
    DI = (
        "extra.dual_imputer.DualImputer",
        {"categorical": categorical}
    )
    OHE = (
        "sklearn.preprocessing.OneHotEncoder",
        {"categorical_features": categorical, "handle_unknown": "ignore", "sparse": False}
    )
    SS = ("sklearn.preprocessing.StandardScaler", {})
    RS = ("sklearn.preprocessing.RobustScaler", {})
    PC = ("sklearn.decomposition.PCA", {})
    PF = ("sklearn.preprocessing.PolynomialFeatures", {})

    # Configuration
    return [
        # Logistic Regression
        {
            "estimator": (
                "sklearn.linear_model.LogisticRegression",
                {"n_jobs": -1, "penalty": "l2", "random_state": random_state}
            ),
            "params": {
                'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
                'dual': [True, False],
                "!@preprocessor": conditional_items([
                    (DI, any_missing, None),
                    ([DI, SS], any_missing, SS)
                ])
            }
        },

        # Random Forests
        {
            "estimator": (
                "sklearn.ensemble.RandomForestClassifier",
                {"n_jobs": -1, "n_estimators": 512, "random_state": random_state}
            ),
            "params": {
                'criterion': ["gini", "entropy"],
                'max_features': np.arange(0.05, 0.5, 0.05).tolist(),
                'max_depth': [8, 9, 10, 11, 12, None],
                'min_samples_split': list(range(2, 21)),
                'min_samples_leaf': list(range(1, 21)),
                'bootstrap': [True, False],
                '!@preprocessor': conditional_items([
                    (DI, any_missing, None),
                    (OHE, not any_missing and any_categorical),
                    ([DI, OHE], any_missing and any_categorical),
                    ([DI, PC], any_missing, PC)
                ])
            }
        },

        # Extra-Random Forests
        {
            "estimator": (
                "sklearn.ensemble.ExtraTreesClassifier",
                {"n_jobs": -1, "n_estimators": 512, "random_state": random_state}
            ),
            "params": {
                'criterion': ["gini", "entropy"],
                'max_features': np.arange(0.05, 0.5, 0.05).tolist(),
                'max_depth': [8, 9, 10, 11, 12, None],
                'min_samples_split': list(range(2, 21)),
                'min_samples_leaf': list(range(1, 21)),
                'bootstrap': [True, False],
                '!@preprocessor': conditional_items([
                    (DI, any_missing, None),
                    (OHE, not any_missing and any_categorical),
                    ([DI, OHE], any_missing and any_categorical),
                    ([DI, PC], any_missing, PC)
                ])
            }
        },

        # C-Support Vector Machine
        {
            "estimator": (
                "sklearn.svm.SVC",
                {"probability": True, "random_state": random_state}
            ),
            "params": {
                "C": np.logspace(-10, 10, num=21, base=2).tolist(),
                "gamma": np.logspace(-10, 0, num=11, base=2).tolist(),
                "kernel": ["linear", "poly", "rbf"],
                "!@preprocessor": conditional_items([
                    (DI, any_missing, None),
                    ([DI, SS], any_missing, SS)
                ])
            }
        },

        # Gradient Boosting
        {
            "estimator": (
                "sklearn.ensemble.GradientBoostingClassifier",
                {"n_estimators": 512, "random_state": random_state}
            ),
            "params": {
                "max_depth": [1, 2, 3],
                "learning_rate": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1],
                "!@preprocessor": conditional_items([
                    (DI, any_missing, None)
                ])
            }
        },

        # Random Tree
        {
            "estimator": (
                "sklearn.tree.ExtraTreeClassifier",
                {"random_state": random_state, "max_depth": None}
            ),
            "params": {
                "criterion": ["gini", "entropy"],
                "max_features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, None],
                "!@preprocessor": conditional_items([
                    (DI, any_missing, None),
                    (OHE, not any_missing and any_categorical),
                    ([DI, OHE], any_missing and any_categorical)
                ])
            }
        },

        # Adaboost
        {
            "estimator": (
                "sklearn.ensemble.AdaBoostClassifier",
                {"random_state": random_state, "!base_estimator": (
                    "sklearn.tree.DecisionTreeClassifier",
                    {}
                )}
            ),
            "params": {
                "learning_rate": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1],
                "algorithm": ["SAMME", "SAMME.R"],
                "n_estimators": [4, 8, 16, 32, 64, 128, 256, 512],
                "!@preprocessor": conditional_items([
                    (DI, any_missing, None)
                ])
            }
        },

        # Multi-layer perceptron
        {
            "estimator": (
                "sklearn.neural_network.MLPClassifier",
                {"max_iter": 500, "random_state": random_state}
            ),
            "params": {
                "activation": ["relu", "tanh", "logistic"],
                "solver": ["lbfgs", "adam"],
                "learning_rate": ['constant', 'adaptive'],
                "hidden_layer_sizes": [(100,), (100, 100), (100, 100, 100)],
                "alpha": [1e-05, 1e-04, 1e-03],
                "!@preprocessor": conditional_items([
                    (None, not any_missing),
                    ([DI, SS], any_missing, SS),
                    ([DI, OHE, SS], any_missing and any_categorical),
                    ([OHE, SS], not any_missing and any_categorical)
                ])
            }
        },

        # Nu-Support Vector Machine
        {
            "estimator": (
                "sklearn.svm.NuSVC",
                {"probability": True, "random_state": random_state}
            ),
            "params": {
                "nu": [0.3, 0.4, 0.5],
                "tol": [0.001, 0.001, 0.01, 0.1],
                "kernel": ["linear", "poly", "rbf"],

                "!@preprocessor": conditional_items([
                    (DI, any_missing, None),
                    ([DI, SS], any_missing, SS)
                ])
            },
        }
    ]


def conditional_items(conditional_steps):
    """Adds items to a list if a condition is fulfilled.

    Parameters
    ----------
    conditional_steps: tuple (item, condition, [alternative])
        A tuple with an item, a boolean condition, and an optional alternative item that will be added instead if the 
        condition does not hold
    Returns
    -------
    The list of items (or alternatives) that fulfill the conditions
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
