from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.pipeline import _name_estimators
from sklearn.utils.metaestimators import if_delegate_has_method


class Pipeline(SklearnPipeline):
    def __init__(self, steps):
        super().__init__(steps)

    @if_delegate_has_method(delegate='_final_estimator')
    def predict(self, X, **kwargs):
        """Apply transforms to the data, and predict with the final estimator

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        Returns
        -------
        y_pred : array-like
        """
        Xt = X
        for name, transform in self.steps[:-1]:
            if transform is not None:
                Xt = transform.transform(Xt)
        return self.steps[-1][-1].predict(Xt, **kwargs)

def make_pipeline(*steps):
    return Pipeline(_name_estimators(steps))