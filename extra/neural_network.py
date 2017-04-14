import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from keras.models import Sequential
from keras.layers import Dense, Dropout


class NeuralNetwork(BaseEstimator, ClassifierMixin):
    def __init__(self, layer_size=100, n_layers=1, activation='relu', optimizer='adam', loss="sparse_categorical_crossentropy",
                 dropout=0.5, readout="softmax", metrics=None, batch_size=30, epochs=5, verbose=0):

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

    def build(self, input_dim, output_dim):

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
            self.model = self.build(input_dim, output_dim)

        self.model.fit(X, y, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose)

        # Return the classifier
        return self

    def predict(self, X):
        # Numpy
        X = np.array(X)

        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        return np.argmax(self.model.predict(X, verbose=self.verbose), axis=1)

    def predict_proba(self, X):
        # Numpy
        X = np.array(X)

        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        return self.model.predict(X, verbose=self.verbose)

    def get_params(self, deep=True):
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
        for parameter, value in parameters.items():
            self.__setattr__(parameter, value)
        return self
