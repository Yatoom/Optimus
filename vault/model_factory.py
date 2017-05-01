import numpy as np


def generate_config(X, categorical):
    # Meta-features to calculate from X
    any_missing = bool(np.isnan(X).any())

    # Pre-processing operators
    DI = (
        "extra.dual_imputer.DualImputer",
        {"categorical": categorical}
    )
    OHE = (
        "sklearn.preprocessing.OneHotEncoder",
        {"categorical_features": categorical, "handle_unknown": "ignore", "sparse": False}
    )
    SS = ("sklearn.preprocessing.StandardScaler", {})
    RS = ("sklearn.preprocessing.RobustScaler", {})
    PC = ("sklearn.decomposition.PCA", {})
    PF = ("sklearn.preprocessing.PolynomialFeatures", {})

    # Configuration
    return [{
        "estimator": (
            "sklearn.linear_model.LogisticRegression",
            {"n_jobs": -1, "penalty": "l2", "random_state": 3}
        ),
        "params": {
            'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
            'dual': [True, False],
            "@preprocessor": conditional_items([
                (DI, any_missing, None),
                ([DI, SS], any_missing, SS)
            ])
        }
    }]


def init_model_config(model_configuration):
    """Initialize model configuration 
    
    Parameters
    ----------
    model_configuration

    Returns
    -------

    """
    configurations = []
    for model in model_configuration:

        if "@preprocessor" in model["params"]:
            model["params"]["@preprocessor"] = source_decode_all(model["params"]["@preprocessor"])

        estimator, init_params = model["estimator"]

        config = {
            "estimator": source_decode(estimator, init_params),
            "params": model["params"]
        }

        configurations.append(config)
    return configurations


def conditional_items(conditional_steps):
    """Adds items to a list if a condition is fulfilled.

    Parameters
    ----------
    conditional_steps: tuple (item, condition, [alternative])
        A tuple with an item, a boolean condition, and an optional alternative item that will be added instead if the 
        condition does not hold
    Returns
    -------
    The list of items (or alternatives) that fulfill the conditions
    """
    result = []
    for conditional_step in conditional_steps:

        if len(conditional_step) == 3:
            step, condition, alternative = conditional_step

            if condition:
                result.append(step)
            else:
                result.append(alternative)

        else:
            step, condition = conditional_step
            if condition:
                result.append(step)

    return result


def source_decode_all(sourcecodes):
    """Decode a list of source paths, recursively 
    
    Parameters
    ----------
    sourcecodes: list or string
        A list of strings of class source (e.g [("sklearn.preprocessing.StandardScaler", {})])

    Returns
    -------

    """
    if isinstance(sourcecodes, list):
        result = []
        for sourcecode in sourcecodes:
            result.append(source_decode_all(sourcecode))
        return result

    if sourcecodes is None:
        return None

    preprocessor, init_params = sourcecodes
    return source_decode(preprocessor, init_params)


def source_decode(sourcecode, init_params):
    """Import class from source path.
    
    Parameters
    ----------
    init_params: dict
        A dictionary with kwargs to pass to created object
    sourcecode: string
        A string of class source (e.g 'sklearn.linear_model.LogisticRegression')
    Returns
    -------
    obj: object
        Class instance (e.g. LogisticRegression(n_jobs=-1, C=100))
    """
    tmp_path = sourcecode.split('.')
    class_name = tmp_path.pop()
    import_path = '.'.join(tmp_path)
    try:
        exec('from {} import {}'.format(import_path, class_name))
        obj = eval(class_name)(**init_params)
    except ImportError:
        print('Warning: {} is not available.'.format(sourcecode))
        obj = None
    return obj
