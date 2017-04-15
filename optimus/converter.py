from copy import copy
import numpy as np


class Converter:
    @staticmethod
    def convert_settings(settings, param_distributions):
        """
        Iteratively calls `convert_settings()` to convert all dictionaries to numerical lists of parameter values.
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
        :param setting: A dictionary with parameter names as keys and parameter values as its values
        :return: A numerical list of parameter values
        """
        settings_copy = copy(setting)

        for key in settings_copy:
            value = settings_copy[key]

            if not isinstance(value, (float, int)):

                # Find the position of the value in the list
                settings_copy[key] = param_distributions[key].index(value)

        return list(settings_copy.values())

    @staticmethod
    def readable_parameters(parameters):
        params = copy(parameters)
        printable = ""

        for key in params:
            printable += key + " = " + str(Converter.make_readable(params[key])) + ", "

        return printable[:-2]

    @staticmethod
    def make_readable(value):
        type_ = type(value)

        if type_ in [int, float, str, bool, np.float64, None]:
            return value

        elif type_ is list or type_ is tuple:
            result = []
            for item in value:
                result.append(Converter.make_readable(item))
            return result

        else:
            return type(value).__name__

    @staticmethod
    def drop_zero_scores(parameters, scores):
        params = copy(parameters)
        mask = (np.array(scores) != 0).tolist()
        params = np.array(params)[mask].tolist()
        scores = np.array(scores)[mask].tolist()
        return params, scores
