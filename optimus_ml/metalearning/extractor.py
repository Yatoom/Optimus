from scipy import stats
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from optimus_ml.extra.stopwatch import Stopwatch


class Extractor:
    @staticmethod
    def calculate(X, y, categorical, cv=10):

        # Calculate stuff that others depend on
        classes = LabelEncoder().fit_transform(y)
        class_occurrences = np.bincount(classes)
        missing = Extractor.missing_values(X)
        kurtoses = Extractor.kurtoses(X, categorical)
        skewnesses = Extractor.skewnesses(X, categorical)
        dt_time_depth1, dt_accuracy_depth1, dt_f1_depth1 = Extractor.landmark_decision_tree(X, y, cv, depth=1)
        dt_time_depth2, dt_accuracy_depth2, dt_f1_depth2 = Extractor.landmark_decision_tree(X, y, cv, depth=2)
        dt_time_depth3, dt_accuracy_depth3, dt_f1_depth3 = Extractor.landmark_decision_tree(X, y, cv, depth=3)
        nb_time, nb_accuracy, nb_f1 = Extractor.landmark_naive_bayes(X, y, cv)
        knn_time, knn_accuracy, knn_f1 = Extractor.landmark_knn(X, y, cv)

        return {
            "instances": X.shape[0],
            "features": X.shape[1],
            "classes": len(class_occurrences),
            "instances with missing values": Extractor.num_instances_with_missing(missing),
            "features with missing values": Extractor.num_features_with_missing(missing),
            "class size min": np.min(class_occurrences),
            "class size max": np.max(class_occurrences),
            "class size mean": np.mean(class_occurrences),
            "class size std": np.std(class_occurrences),
            "categorical ratio": np.sum(categorical) / len(categorical),
            "kurtosis min": np.nanmin(kurtoses),
            "kurtosis max": np.nanmax(kurtoses),
            "kurtosis mean": np.nanmean(kurtoses),
            "kurtosis std": np.nanstd(kurtoses),
            "skewness min": np.nanmin(skewnesses),
            "skewness max": np.nanmax(skewnesses),
            "skewness mean": np.nanmean(skewnesses),
            "skewness std": np.nanstd(skewnesses),
            "decision tree time depth 1": dt_time_depth1,
            "decision tree accuracy depth 1": dt_accuracy_depth1,
            "decision tree f1 depth 1": dt_f1_depth1,
            "decision tree time depth 2": dt_time_depth2,
            "decision tree accuracy depth 2": dt_accuracy_depth2,
            "decision tree f1 depth 2": dt_f1_depth2,
            "decision tree time depth 3": dt_time_depth3,
            "decision tree accuracy depth 3": dt_accuracy_depth3,
            "decision tree f1 depth 3": dt_f1_depth3,
            "naive bayes time": nb_time,
            "naive bayes accuracy": nb_accuracy,
            "naive bayes f1": nb_f1,
            "knn time": knn_time,
            "knn accuracy": knn_accuracy,
            "knn f1": knn_f1
        }

    @staticmethod
    def missing_values(X):
        return ~np.isfinite(X)

    @staticmethod
    def num_instances_with_missing(missing):
        num_missing = missing.sum(axis=1)
        bin_missing = [1 if num > 0 else 0 for num in num_missing]
        return float(np.sum(bin_missing))

    @staticmethod
    def num_features_with_missing(missing):
        num_missing = missing.sum(axis=0)
        bin_missing = [1 if num > 0 else 0 for num in num_missing]
        return float(np.sum(bin_missing))

    @staticmethod
    def kurtoses(X, categorical):
        if np.array(categorical).all():
            return [0]

        kurts = []
        for i in range(X.shape[1]):
            if not categorical[i]:
                kurts.append(stats.kurtosis(X[:, i]))
        return kurts

    @staticmethod
    def skewnesses(X, categorical):
        if np.array(categorical).all():
            return [0]

        skews = []
        for i in range(X.shape[1]):
            if not categorical[i]:
                skews.append(stats.skew(X[:, i]))
        return skews

    @staticmethod
    def landmark(estimator, X, y, cv):
        with Stopwatch() as sw:
            y_pred = cross_val_predict(estimator=estimator, X=X, y=y, cv=cv, n_jobs=-1)
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average="weighted")

        return sw.duration, accuracy, f1

    @staticmethod
    def landmark_naive_bayes(X, y, cv):
        clf = GaussianNB()
        duration, accuracy, f1 = Extractor.landmark(clf, X, y, cv)
        return duration, accuracy, f1

    @staticmethod
    def landmark_decision_tree(X, y, cv, depth=1):
        clf = DecisionTreeClassifier(max_depth=depth)
        duration, accuracy, f1 = Extractor.landmark(clf, X, y, cv)
        return duration, accuracy, f1

    @staticmethod
    def landmark_knn(X, y, cv):
        clf = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
        duration, accuracy, f1 = Extractor.landmark(clf, X, y, cv)
        return duration, accuracy, f1
