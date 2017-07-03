from copy import copy
import numpy as np


class Converter:
    @staticmethod
    def convert_settings(settings, param_distributions, robust=False):
        """
        Iteratively calls `convert_settings()` to convert all settings to its numerical representations.

        Parameters
        ----------
        settings: list
            A list of dictionaries with parameter names as keys and parameter values as its values

        param_distributions: dict
            Dictionary of parameter distributions

        robust: bool
            Whether to use `convert_settings()` or `convert_settings_robust()`

        Returns
        -------
        A list with a numerical representation for each setting
        """

        result = []
        for setting in settings:
            if robust:
                result.append(Converter.convert_setting_robust(setting, param_distributions))
            else:
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
    def convert_setting_robust(setting, param_distributions):
        """
        Converts a setting to numbers, and automatically resolves issues where keys or values from the setting
        do not agree with the param distribution.

        For each parameter key in param_distributions, we check if the key appears in the setting, and if the
        corresponding parameter value appears in the list of possible values from param_distributions. We resolve three
        cases.

        1) Key not in setting
            We take the median index.

        2) Key in setting, but value not in possible values
            If the value is a number, we try to get the index of the closest number in the possible values.
            If the value is not a number, we take the median index.

        3) Key in setting and value in possible values
            We take the index of the value.

        Parameters
        ----------
        setting: dict
            A dictionary with parameter names as keys and parameter values as its values

        param_distributions: dict
            Dictionary of parameter distributions (lists)

        Returns
        -------
        Returns a numerical representation of the setting.
        """
        result = []

        for key, possible_values in param_distributions.items():

            # case 1: key not in setting
            if key not in setting:
                median_index = (len(possible_values) - 1) / 2
                result.append(median_index)

            # case 2: key in setting but value not in possible_values
            elif setting[key] not in possible_values:

                # If it is a number, try to find closest number
                if isinstance(setting[key], (np.int_, np.float_, float, int)):
                    closest = possible_values[0]
                    shortest_distance = np.inf
                    for i in possible_values:
                        if isinstance(i, (np.int_, np.float_, float, int)):
                            distance = np.abs(i - setting[key])
                            if distance < shortest_distance:
                                closest = i
                                shortest_distance = distance
                    index_closest = possible_values.index(closest)
                    result.append(index_closest)
                else:
                    median_index = (len(possible_values) - 1) / 2
                    result.append(median_index)

            # case 3: key in setting and value in possible_values
            else:
                index = possible_values.index(setting[key])
                result.append(index)

        return result

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
    def readable_dict(dict_):
        """
        Makes a dictionary readable, so it can be stored as JSON.

        Parameters
        ----------
        dict_: dict
            Mapping of parameter keys to values

        Returns
        -------
        Returns a dictionary with human-readable values.
        """
        dict_copy = copy(dict_)
        for key, item in dict_.items():
            dict_copy[key] = Converter.make_readable(item)
        return dict_copy

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