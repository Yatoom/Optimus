from copy import copy
import numpy as np


class Converter:
    @staticmethod
    def convert_settings(settings, param_distributions):
        """
        Iteratively calls `convert_settings()` to convert all dictionaries to numerical lists of parameter values.
        :param param_distributions: Dictionary of parameter distributions
        :param settings: A list of dictionaries with parameter names as keys and parameter values as its values
        :return: A list of numerical lists of parameter values
        """
        result = []
        for setting in settings:
            result.append(Converter.convert_setting(setting, param_distributions))
        return result

    @staticmethod
    def convert_setting(setting, param_distributions):
        """
        Takes the values of a parameter dictionary and converts them to numbers if necessary.
        :param param_distributions: Dictionary of parameter distributions
        :param setting: A dictionary with parameter names as keys and parameter values as its values
        :return: A numerical list of parameter values
        """
        settings_copy = copy(setting)

        for key in settings_copy:
            value = settings_copy[key]

            # Find the position of the value in the list
            settings_copy[key] = param_distributions[key].index(value)

        return list(settings_copy.values())

    @staticmethod
    def readable_parameters(parameters):
        """
        Converts a dictionary of parameters to a readable string.
        :param parameters: A dictionary of parameters
        :return: A readable string of parameters and their values
        """
        params = copy(parameters)
        printable = ""

        for key in params:
            printable += str(key) + " = " + str(Converter.make_readable(params[key])) + ", "

        return printable[:-2]

    @staticmethod
    def make_readable(value):
        """
        Makes a value more readable for humans by converting objects to names.
        :param value: Any type of value
        :return: The original int, float, str, bool, np.float64 or NoneType, or the name of the value otherwise, and in 
        case of a list or tuple, it will return a list with converted values.
        """
        type_ = type(value)

        if type_ in [int, float, str, bool, np.float64, type(None)]:
            return value

        elif type_ is list or type_ is tuple:
            result = []
            for item in value:
                result.append(Converter.make_readable(item))
            return result

        else:
            return type(value).__name__

    @staticmethod
    def remove_timeouts(parameters, scores, timeout_score=0):
        """
        Removes the validations that got a timeout. These are useful to keep to direct the Gaussian Process away from
        these points, but if we need a realistic estimation of the expected improvement, we should remove these points.
        :param timeout_score: The score value that indicates a timeout
        :param parameters: A list of all validated parameters
        :param scores: A list of all validation scores
        :return: The new (parameters, scores) without timeouts
        """
        params = copy(parameters)
        mask = (np.array(scores) != timeout_score).tolist()
        params = np.array(params)[mask].tolist()
        scores = np.array(scores)[mask].tolist()
        return params, scores
