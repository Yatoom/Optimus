import copy
import warnings
import numpy as np


def grid_to_json(grid):
    """
    Make a grid JSON serializable.

    Parameters
    ----------
    grid: dict
        A grid to convert

    Returns
    -------
    The JSON serializable grid
    """
    converted = {}

    for key, params in grid.items():
        result = []
        for item in params:
            result.append(_value_to_json(item))
        converted[key] = result

    return converted


def reconstruct_grid(o):
    """
    Reconstruct the original from a JSON serializable grid.

    Parameters
    ----------
    o: dict
        The object from which to reconstruct

    Returns
    -------
    The reconstructed grid
    """
    converted = {}

    for key, params in o.items():
        result = []
        for item in params:
            result.append(_reconstruct_value(item))
        converted[key] = result

    return converted


def settings_to_json(settings):
    """
    Make settings JSON serializable.

    Parameters
    ----------
    settings: list
        The settings to convert

    Returns
    -------
    The converted settings
    """

    result = []

    for setting in settings:
        result.append(setting_to_json(setting))

    return result


def reconstruct_settings(o):
    """
    Reconstruct settings from JSON serializable object.

    Parameters
    ----------
    o: list
         The object from which to reconstruct

    Returns
    -------
    The reconstructed settings
    """

    result = []

    for setting in o:
        result.append(reconstruct_setting(setting))

    return result


def setting_to_json(setting):
    """
    Make a setting JSON serializable.

    Parameters
    ----------
    setting: dict
        A setting to convert

    Returns
    -------
    The converted setting
    """
    converted = {}

    for key, value in setting.items():
        converted[key] = _value_to_json(value)

    return converted


def reconstruct_setting(o):
    """
    Reconstruct the original setting from a JSON serializable object.

    Parameters
    ----------
    o: dict
        The object from which to reconstruct

    Returns
    -------
    The reconstructed setting
    """
    converted = {}

    for key, value in o.items():
        converted[key] = _reconstruct_value(value)

    return converted


def settings_to_indices(settings, param_distributions, robust=False):
    result = []
    for setting in settings:
        if robust:
            result.append(setting_to_indices_robust(setting, param_distributions))
        else:
            result.append(setting_to_indices(setting, param_distributions))
    return result


def setting_to_indices(setting, param_distributions):
    """
    Transforms a setting dictionary to a numerical list, so that we can fit and predict with a regressor

    Parameters
    ----------
    setting: dict
        Dictionary with parameter names as keys and parameter values as its values

    param_distributions: dict
        Dictionary of parameter distributions

    Returns
    -------
    A numerical list of parameter indices
    """

    settings_copy = copy.copy(setting)

    for key in settings_copy:
        value = settings_copy[key]

        # Find the position of the value in the list
        settings_copy[key] = param_distributions[key].index(value)

    return list(settings_copy.values())


def setting_to_indices_robust(setting, param_distributions):
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


def dictionary_to_readable_dictionary(dictionary):
    """
    For each key, convert its value to a readable format.

    Parameters
    ----------
    dictionary: dict
        The dictionary to make readable

    Returns
    -------
    The dictionary with human-readable values.
    """
    dictionary_copy = copy.copy(dictionary)

    for key, item in dictionary.items():
        dictionary_copy[key] = value_to_readable(item)

    return dictionary_copy


def dictionary_to_string(dictionary):
    """
    Convert a dictionary to a string representation.

    Parameters
    ----------
    dictionary: dict
        The dictionary to make readable

    Returns
    -------
    A string representation of the dictionary
    """

    printable = ""

    for key, value in dictionary.items():
        printable += str(key) + " = " + str(value_to_readable(value)) + ", "

    return printable[:-2]


def value_to_readable(value):
    """
    Makes a value more readable for humans by converting objects to names.

    Parameters
    ----------
    value: any type
        The value to convert

    Returns
    -------
    The original int, float, str, bool, np.float64 or NoneType, or the name of the value otherwise, and in
    case of a list or tuple, it will return a list with converted values.
    """

    if isinstance(value, (str, int, float, bool, np.int_, np.float)):
        return value

    elif isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.append(value_to_readable(item))
        return result

    return type(value).__name__


def _value_to_json(value):
    """
    Make a value JSON serializable.

    Parameters
    ----------
    value: primitive or object
        The value to convert

    Returns
    -------
    A JSON serializable value, or None
    """
    if isinstance(value, (str, int, float, bool, np.int_, np.float)):
        return value

    elif hasattr(value, "get_params"):
        return {
            "source": "{}.{}".format(type(value).__module__, type(value).__name__),
            "params": value.get_params()
        }

    return None


def _reconstruct_value(o):
    """
    Reconstruct the original value from a JSON serializable value.

    Parameters
    ----------
    o: JSON serializable value
        The value from which to reconstruct

    Returns
    -------
    The reconstructed value
    """
    if isinstance(o, (str, int, float, bool, np.int_, np.float)):
        return o

    elif isinstance(o, dict) and "source" in o and "params" in o:
        return _decode_source(o["source"], **o["params"])

    return None


def _decode_source(source, **init_params):
    """
    Decode a source string (for example 'sklearn.preprocessing.data.StandardScaler') to an object and initialize with
    given parameters.

    Parameters
    ----------
    source: string
        The source to the class
    init_params: **kwargs
        Keyword arguments to initialize the object with
    Returns
    -------
    The object instance
    """
    path_parts = source.split('.')
    class_name = path_parts.pop()
    import_path = '.'.join(path_parts)

    try:
        exec('from {} import {}'.format(import_path, class_name))
        class_type = eval(class_name)(**init_params)
    except ImportError:
        warnings.warn('Warning: {} is not available.'.format(source))
        class_type = None
    return class_type
