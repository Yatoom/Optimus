from copy import copy

from sklearn import clone
from sklearn.pipeline import make_pipeline


class Builder:
    @staticmethod
    def build_pipeline(base_estimator, parameters):
        """
        Builds a pipeline where the base estimator is initialized with given parameters. The `@preprocessor` parameter
        is a special parameter that will determine which pre-processing steps to use.
        :param base_estimator: The base estimator of the pipeline
        :param parameters: The parameters for the base estimator, includes special parameters for the pipeline itself
        :return: The (pipeline with the) base estimator, initialized with given parameters
        """
        params = copy(parameters)
        preprocessors = Builder.extract_preprocessors(params)
        estimator = Builder.setup_estimator(base_estimator, params)

        if preprocessors is None:
            return estimator

        return make_pipeline(*preprocessors, estimator)

    @staticmethod
    def extract_preprocessors(params):
        """
        Removes the `@preprocessor` keyword from the parameter dictionary and returns the list of preprocessor steps. 
        :param params: The parameter dictionary
        :return: The preprocessor steps
        """
        preprocessors = None

        if "@preprocessor" in params:

            preprocessors = params["@preprocessor"]

            if type(preprocessors) is not list:
                preprocessors = [preprocessors]

            params.pop("@preprocessor")

        return preprocessors

    @staticmethod
    def setup_estimator(base_estimator, parameters):
        """
        Sets the parameters of the base estimator. 
        :param base_estimator: The base estimator
        :param parameters: Dictionary of parameters to set
        :return: A clone of the base estimators with given parameters
        """
        return clone(base_estimator).set_params(**parameters)
