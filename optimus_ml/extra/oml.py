from openml import runs
import sklearn
from optimus_ml.optimizer.builder import Builder
from optimus_ml import converter


def initialize_model_from_trace(run_id, repeat, fold, iteration=None):
    '''
    Initialize a model based on the parameters that were set
    by an optimization procedure (i.e., using the exact same
    parameter settings)

    Parameters
    ----------
    run_id : int
        The Openml run_id. Should contain a trace file,
        otherwise a OpenMLServerException is raised

    repeat: int
        The repeat nr (column in trace file)

    fold: int
        The fold nr (column in trace file)

    iteration: int
        The iteration nr (column in trace file). If None, the
        best (selected) iteration will be searched (slow),
        according to the selection criteria implemented in
        OpenMLRunTrace.get_selected_iteration

    Returns
    -------
    model : sklearn model
        the scikit-learn model with all parameters initailized
    '''
    run_trace = runs.get_run_trace(run_id)

    if iteration is None:
        iteration = run_trace.get_selected_iteration(repeat, fold)

    request = (repeat, fold, iteration)
    if request not in run_trace.trace_iterations:
        raise ValueError('Combination repeat, fold, iteration not availavle')
    current = run_trace.trace_iterations[(repeat, fold, iteration)]

    search_model = runs.initialize_model_from_run(run_id)
    if not isinstance(search_model, sklearn.model_selection._search.BaseSearchCV):
        raise ValueError('Deserialized flow not instance of ' \
                         'sklearn.model_selection._search.BaseSearchCV')
    base_estimator = search_model.estimator

    params = current.get_parameters()

    for key, value in params.items():
        if isinstance(value, str) and "source" in value and "params" in value:
            val = eval(value)
            params[key] = converter.reconstruct_value(val)

    pipeline = Builder.build_pipeline(base_estimator, params)

    return pipeline
