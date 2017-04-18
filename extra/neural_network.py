import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from keras.models import Sequential
from keras.layers import Dense, Dropout


class NeuralNetwork(BaseEstimator, ClassifierMixin):
    def __init__(self, layer_size=100, n_layers=1, activation='relu', optimizer='adam', loss="sparse_categorical_crossentropy",
                 dropout=0.5, readout="softmax", metrics=None, batch_size=30, epochs=5, verbose=0):
        """
        Keras implementation of a simple neural network.
        :param layer_size: Integer number of nodes per layer 
        :param n_layers: Integer number of layers
        :param activation: Keras activation function
        :param optimizer: Keras optimizer
        :param loss: kerass loss function
        :param dropout: Droupout ratio
        :param readout: Keras activation function for the readout layer
        :param metrics: List of Keras metrics to evaluate each round
        :param batch_size: Integer number for batch size
        :param epochs: Integer number of epochs
        :param verbose: Verbosity level
        """

        if metrics is None:
            metrics = ["accuracy"]

        self.layer_size = layer_size
        self.n_layers = n_layers
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.dropout = dropout
        self.readout = readout
        self.metrics = metrics
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose

        self.io = (None, None)
        self.classes_ = None
        self.model = None

    def _build(self, input_dim, output_dim):
        model = Sequential()

        # First hidden layer
        model.add(Dense(self.layer_size, activation=self.activation, input_dim=input_dim))
        model.add(Dropout(self.dropout))

        # All other hidden layers
        for i in range(0, self.n_layers - 1):
            model.add(Dense(self.layer_size, activation=self.activation))
            model.add(Dropout(self.dropout))

        # Read-out layer
        model.add(Dense(output_dim, activation=self.readout))

        # Compile the model
        model.compile(loss=self.loss,
                      optimizer=self.optimizer,
                      metrics=self.metrics)

        return model

    def fit(self, X, y):
        """
        Fit on X.
        :param X: {array-like, sparse matrix}, shape (n_samples, n_features). Input data, where `n_samples` is the 
        number of samples and `n_features` is the number of features.
        :return: Returns self
        """

        # Numpy
        X = np.array(X)
        y = np.array(y)

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        # Store so that we know what we fitted on
        self.X_ = X
        self.y_ = y

        # Get dimensions
        input_dim = X.shape[1]
        output_dim = len(self.classes_)

        # Create a model if needed
        if (input_dim, output_dim) != self.io:
            self.model = self._build(input_dim, output_dim)

        self.model.fit(X, y, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose)

        # Return the classifier
        return self

    def predict(self, X):
        """
        Predict class value for X.
        :param X: {array-like, sparse matrix}, shape (n_samples, n_features). Input data, where `n_samples` is the 
        number of samples and `n_features` is the number of features.
        :return: Returns self. 
        """

        # Numpy
        X = np.array(X)

        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        return np.argmax(self.model.predict(X, verbose=self.verbose), axis=1)

    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        :param X: {array-like, sparse matrix}, shape (n_samples, n_features). Input data, where `n_samples` is the 
        number of samples and `n_features` is the number of features.
        :return: Returns self. 
        """

        # Numpy
        X = np.array(X)

        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        return self.model.predict_proba(X, verbose=self.verbose)

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        :param deep: boolean, optional. If True, will return the parameters for this estimator and contained subobjects 
        that are estimators.
        :return: Parameter names mapped to their values.
        """
        return {
            "layer_size": self.layer_size,
            "activation": self.activation,
            "optimizer": self.optimizer,
            "loss": self.loss,
            "dropout": self.dropout,
            "readout": self.readout,
            "metrics": self.metrics,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "verbose": self.verbose
        }

    def set_params(self, **parameters):
        """
        Set the parameters of this estimator.
        :param parameters: Parameter-value mapping of parameters to set
        :return: Returns self
        """
        for parameter, value in parameters.items():
            self.__setattr__(parameter, value)
        return self
