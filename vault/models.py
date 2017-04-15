import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from extra.dual_imputer import DualImputer


def get_models(categorical):
    # Pre-processing
    DI = DualImputer(categorical=categorical)
    OHE = OneHotEncoder(categorical_features=categorical, handle_unknown="ignore", sparse=False)
    SS = StandardScaler()
    RS = RobustScaler()
    PC = PCA()
    PF = PolynomialFeatures()

    models = [
        {
            "name": "Random Forest",
            "estimator": RandomForestClassifier(n_jobs=-1, n_estimators=300, random_state=3),
            "params": {
                'criterion': ["gini", "entropy"],
                'max_features': np.arange(0.05, 0.5, 0.05),
                'max_depth': [3, 5, 7, 9, None],
                'min_samples_split': range(2, 21),
                'min_samples_leaf': range(1, 21),
                'bootstrap': [True, False],
                '@preprocessor': [DI, [DI, OHE]]
            }
        },
        {
            "name": "SVM Kernels",
            "estimator": SVC(probability=True),
            "params": {
                "C": np.logspace(-10, 10, num=21, base=2),
                "gamma": np.logspace(-10, 0, num=11, base=2),
                "kernel": ["linear", "poly", "rbf"],
                "@preprocessor": [DI, [DI, SS]]
            }
        },
        {
            "name": "Gradient Boosting",
            "estimator": GradientBoostingClassifier(random_state=3, n_estimators=512),
            "params": {
                "max_depth": [1, 2, 3],
                "learning_rate": np.logspace(-5, 0, num=15, base=10),
                "@preprocessor": [DI]
            }
        },
        {
            "name": "K-Neighbors",
            "estimator": KNeighborsClassifier(n_jobs=-1),
            "params": {
                'n_neighbors': [1, 3, 5, 7, 9],
                'weights': ["uniform", "distance"],
                'p': [1, 2],
                "@preprocessor": [DI, [DI, SS]]
            }
        },
        {
            "name": "LogisticRegression",
            "estimator": LogisticRegression(n_jobs=-1, penalty="l2", random_state=3),
            "params": {
                'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
                'dual': [True, False],
                "@preprocessor": [DI, [DI, SS]]
            }
        },
    ]

    return models
