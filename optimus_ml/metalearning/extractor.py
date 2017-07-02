import asyncio

from scipy import stats
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from optimus_ml.extra.dual_imputer import DualImputer
from optimus_ml.extra.stopwatch import Stopwatch


class Extractor:

    result = {}

    @staticmethod
    def calculate(X, y, categorical, cv=10):

        # Calculate stuff in parallel that others depend on
        loop = asyncio.get_event_loop()
        loop.run_until_complete(asyncio.gather(
            Extractor.simple_stats(X, categorical),
            Extractor.landmark_knn(X, y, cv, categorical),
            Extractor.landmark_naive_bayes(X, y, cv, categorical),
            Extractor.landmark_decision_tree(X, y, cv, categorical, depth=1),
            Extractor.landmark_decision_tree(X, y, cv, categorical, depth=2),
            Extractor.landmark_decision_tree(X, y, cv, categorical, depth=3),
            Extractor.kurtoses(X, categorical),
            Extractor.skewnesses(X, categorical),
            Extractor.missing_values(X),
            Extractor.classes(y)
        ))
        loop.close()
        return Extractor.result

    @staticmethod
    async def simple_stats(X, categorical):
        Extractor.result["instances"] = X.shape[0]
        Extractor.result["features"] = X.shape[1]
        Extractor.result["categorical ratio"] = np.sum(categorical) / len(categorical)

    @staticmethod
    async def missing_values(X):
        missing = ~np.isfinite(X)
        num_missing_per_instance = missing.sum(axis=1)
        num_missing_per_feature = missing.sum(axis=0)

        Extractor.result["number of instances with missing"] = np.sum(np.count_nonzero(num_missing_per_instance))
        Extractor.result["number of features with missing"] = np.sum(np.count_nonzero(num_missing_per_feature))

    @staticmethod
    async def kurtoses(X, categorical):
        if np.array(categorical).all():
            return [0]

        kurts = []
        for i in range(X.shape[1]):
            if not categorical[i]:
                kurts.append(stats.kurtosis(X[:, i]))

        Extractor.result["kurtosis min"] = np.nanmin(kurts)
        Extractor.result["kurtosis max"] = np.nanmax(kurts)
        Extractor.result["kurtosis mean"] = np.nanmean(kurts)
        Extractor.result["kurtosis std"] = np.nanstd(kurts)

    @staticmethod
    async def skewnesses(X, categorical):
        if np.array(categorical).all():
            return [0]

        skews = []
        for i in range(X.shape[1]):
            if not categorical[i]:
                skews.append(stats.skew(X[:, i]))

        # Extractor.result["skews"] = skews
        Extractor.result["skews min"] = np.nanmin(skews)
        Extractor.result["skews max"] = np.nanmax(skews)
        Extractor.result["skews mean"] = np.nanmean(skews)
        Extractor.result["skews std"] = np.nanstd(skews)

    @staticmethod
    async def landmark(estimator, X, y, cv, categorical, name=None):

        pipeline = make_pipeline(DualImputer(categorical=categorical), estimator)

        with Stopwatch() as sw:
            y_pred = cross_val_predict(estimator=pipeline, X=X, y=y, cv=cv, n_jobs=-1)

        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average="weighted")

        if name is None:
            name = type(estimator).__name__

        Extractor.result["{} time".format(name)] = sw.duration
        Extractor.result["{} accuracy".format(name)] = accuracy
        Extractor.result["{} f1".format(name)] = f1

    @staticmethod
    async def landmark_naive_bayes(X, y, cv, categorical):
        clf = GaussianNB()
        await Extractor.landmark(clf, X, y, cv, categorical)

    @staticmethod
    async def landmark_decision_tree(X, y, cv, categorical, depth=1):
        clf = DecisionTreeClassifier(max_depth=depth)
        await Extractor.landmark(clf, X, y, cv, categorical, name="{} depth {}".format(type(clf).__name__, depth))

    @staticmethod
    async def landmark_knn(X, y, cv, categorical):
        clf = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
        await Extractor.landmark(clf, X, y, cv, categorical)

    @staticmethod
    async def classes(y):
        classes = LabelEncoder().fit_transform(y)
        occurrences = np.bincount(classes)
        Extractor.result["number of classes"] = len(occurrences)
        Extractor.result["class size mean"] = np.nanmean(occurrences)
        Extractor.result["class size min"] = np.nanmin(occurrences)
        Extractor.result["class size max"] = np.nanmax(occurrences)
        Extractor.result["class size std"] = np.nanstd(occurrences)