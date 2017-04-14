import numpy as np
from sklearn.preprocessing import Imputer


class DualImputer:
    def __init__(self, strategy_categorical="most_frequent", strategy_numerical="median", categorical=None):
        if categorical is None:
            categorical = []
        self.strategy_categorical = strategy_categorical
        self.strategy_numerical = strategy_numerical
        self.cat_imputer = Imputer(strategy=strategy_categorical)
        self.num_imputer = Imputer(strategy=strategy_numerical)
        self.categorical = categorical
        self._update_indices()

    def _update_indices(self):
        self.numerical = np.logical_not(self.categorical)
        self.cat_indices = np.where(self.categorical)[0]
        self.num_indices = np.where(self.numerical)[0]

    def fit(self, X, y=None):
        cat_features, num_features = self._split(X)

        if cat_features.shape[1] > 0:
            self.cat_imputer.fit(cat_features)

        if num_features.shape[1] > 0:
            self.num_imputer.fit(num_features)
        return self

    def fit_transform(self, X, y=None):
        cat_features, num_features = self._split(X)

        if cat_features.shape[1] > 0:
            cat_features = self.cat_imputer.fit_transform(cat_features)

        if num_features.shape[1] > 0:
            num_features = self.num_imputer.fit_transform(num_features)

        return self._combine(cat_features, num_features)

    def transform(self, X, y=None):
        cat_features, num_features = self._split(X)

        if cat_features.shape[1] > 0:
            cat_features = self.cat_imputer.transform(cat_features)

        if num_features.shape[1] > 0:
            num_features = self.num_imputer.transform(num_features)

        return self._combine(cat_features, num_features)

    def _split(self, X):
        cat_features = X.compress(self.categorical, axis=1)
        num_features = X.compress(self.numerical, axis=1)
        return cat_features, num_features

    def _combine(self, included, excluded):
        combined = np.hstack((included, excluded))
        order = self._reverse_index(np.append(self.cat_indices, self.num_indices))
        return combined[:, order]

    @staticmethod
    # Swaps the indices of a list with its values
    def _reverse_index(order):
        result = [-1 for _ in range(0, len(order))]
        for index, old_index in enumerate(order):
            result[old_index] = index

        return result

    def get_params(self, deep=False):
        return {
            "strategy_categorical": self.strategy_categorical,
            "strategy_numerical": self.strategy_numerical,
            "categorical": self.categorical
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.__setattr__(parameter, value)

        # Update parameters that are dependent on others.
        self._update_indices()
        return self
