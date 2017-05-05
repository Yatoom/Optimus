import warnings
from copy import copy


def encode_object(o):
    """
    Encode an object instance to a source string (for example 'sklearn.preprocessing.data.StandardScaler')
    
    Parameters
    ----------
    o: object
        The object to encode
    Returns
    -------
    A source string to the object
    """
    return "{}.{}".format(type(o).__module__, type(o).__name__)


def decode_source(source, **init_params):
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


def decode_sources(sources_with_params, special_prefix="!"):
    """
    Wrapper around `decode_source()` that accepts (a list of) tuples of kind (source, params).
    
    Parameters
    ----------
    special_prefix:
        Prefix for special parameters that need decoding
    sources_with_params: {list, tuple}
        A tuple or a list of tuples of the shape (str, dict)
    Returns
    -------
    The initialized object instance(s)
    """

    # If input is a list, we recursively call this function
    if isinstance(sources_with_params, list):
        result = []
        for source_with_params in sources_with_params:
            result.append(decode_sources(source_with_params))
        return result

    if isinstance(sources_with_params, tuple):
        source, params = sources_with_params
        decoded_params = decode_params(params, prefix=special_prefix)

        return decode_source(source, **decoded_params)

    # We might have found an object that doesn't need decoding
    return sources_with_params


def decode_params(params, prefix="!", remove_prefixes=True):
    """
    Check each parameter and if the parameter starts with a given sign, we will assume its value is a list of tuples
    (source, params) that need to be source decoded and initialized with given parameters.
    
    Parameters
    ----------
    params: dict
        A dictionary of parameters
    prefix: str
        A prefix that indicates the parameter is special and requires decoding
    remove_prefixes: bool
        Indicates if prefixes should be removed from keys
    Returns
    -------
    Returns the new parameters where the key prefixes are removed and its values are decoded  
    """

    # Make a copy
    params_copy = copy(params)

    for key in params:

        # Check if key starts with prefix
        if key[0:len(prefix)] == prefix:

            # Decode value
            decoded = decode_sources(params_copy[key])

            # Store decoded value
            if remove_prefixes:
                params_copy[key[len(prefix):]] = decoded
                del params_copy[key]
            else:
                params_copy[key] = decoded

    return params_copy
