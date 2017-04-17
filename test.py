from sklearn.preprocessing import OneHotEncoder

from extra.dual_imputer import DualImputer
from optimus.converter import Converter
import numpy as np

DI = DualImputer(categorical=[])
OHE = OneHotEncoder(categorical_features=[], handle_unknown="ignore", sparse=False)

param_dist = {
    'criterion': ["gini", "entropy"],
    'max_features': np.arange(0.05, 0.5, 0.05),
    'max_depth': [3, 5, 7, 9, None],
    'min_samples_split': range(2, 21),
    'min_samples_leaf': range(1, 21),
    'bootstrap': [True, False],
    '@preprocessor': [DI, [DI, OHE]]
}

setting = {"criterion": "entropy", "max_features": 0.05, "max_depth": 3, "min_samples_split": 2, "min_samples_leaf": 2, "bootstrap": True, '@preprocessor': DI}

print(Converter.convert_setting(setting, param_dist))