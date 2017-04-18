import numpy as np
from sklearn.preprocessing import Imputer


class DualImputer:
    def __init__(self, strategy_categorical="most_frequent", strategy_numerical="median", categorical=None):
        """
        An Imputer that can apply a different strategy for both categorical data and numerical data.
        :param strategy_categorical: "mean", "median" or "most_frequent"
        :param strategy_numerical: "mean", "median" or "most_frequent"
        :param categorical: A boolean mask for the categorical columns of a dataset
        """
        if categorical is None:
            categorical = []
        self.strategy_categorical = strategy_categorical
        self.strategy_numerical = strategy_numerical
        self.cat_imputer = Imputer(strategy=strategy_categorical)
        self.num_imputer = Imputer(strategy=strategy_numerical)
        self.categorical = categorical
        self._update_indices()

    def fit(self, X, y=None):
        """
        Fit the imputer on X.
        :param X: {array-like, sparse matrix}, shape (n_samples, n_features). Input data, where `n_samples` is the 
        number of samples and `n_features` is the number of features.
        :return: Returns self
        """
        cat_features, num_features = self._split(X)

        if cat_features.shape[1] > 0:
            self.cat_imputer.fit(cat_features)

        if num_features.shape[1] > 0:
            self.num_imputer.fit(num_features)
        return self

    def fit_transform(self, X, y=None):
        """
        Call fit and transform on X.
        :param X: {array-like, sparse matrix}, shape (n_samples, n_features). Input data, where `n_samples` is the 
        number of samples and `n_features` is the number of features.
        :return: Transformed X 
        """
        cat_features, num_features = self._split(X)

        if cat_features.shape[1] > 0:
            cat_features = self.cat_imputer.fit_transform(cat_features)

        if num_features.shape[1] > 0:
            num_features = self.num_imputer.fit_transform(num_features)

        return self._combine(cat_features, num_features)

    def transform(self, X, y=None):
        """
        Impute all missing values in X.
        :param X: {array-like, sparse matrix}, shape (n_samples, n_features). Input data, where `n_samples` is the 
        number of samples and `n_features` is the number of features.
        :return: Transformed X
        """
        cat_features, num_features = self._split(X)

        if cat_features.shape[1] > 0:
            cat_features = self.cat_imputer.transform(cat_features)

        if num_features.shape[1] > 0:
            num_features = self.num_imputer.transform(num_features)

        return self._combine(cat_features, num_features)

    def get_params(self, deep=False):
        """
        Get parameters for this estimator.
        :param deep: boolean, optional. If True, will return the parameters for this estimator and contained subobjects 
        that are estimators.
        :return: Parameter names mapped to their values.
        """
        return {
            "strategy_categorical": self.strategy_categorical,
            "strategy_numerical": self.strategy_numerical,
            "categorical": self.categorical
        }

    def set_params(self, **parameters):
        """
        Set the parameters of this estimator.
        :param parameters: Parameter-value mapping of parameters to set
        :return: Returns self
        """

        for parameter, value in parameters.items():
            self.__setattr__(parameter, value)

        # Update parameters that are dependent on others.
        self._update_indices()
        return self

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

    def _update_indices(self):
        self.numerical = np.logical_not(self.categorical)
        self.cat_indices = np.where(self.categorical)[0]
        self.num_indices = np.where(self.numerical)[0]