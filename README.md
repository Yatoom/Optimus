<p align="center">
    <img src="http://jeroenvanhoof.nl/optimus2.svg" width="50%"/>
</p>

# Optimus
An automated machine learning tool.

## Installation
```bash
pip install optimus-ml
```

## Example usage
### Using Optimus without OpenML
```python
from optimus_ml import ModelOptimizer
from sklearn.svm import SVC
from sklearn.datasets import load_iris
import numpy as np

# Load data
data = load_iris()
X = data.data
y = data.target

# Setup classifier
clf = SVC(probability=True, random_state=3)

# Setup parameter grid
param_grid = {
    "C": np.logspace(-10, 10, num=21, base=2).tolist(),
    "gamma": np.logspace(-10, 0, num=11, base=2).tolist(),
    "kernel": ["linear", "poly", "rbf"],
}

# Setup Model Optimizer
opt = ModelOptimizer(estimator=clf, encoded_params=param_grid, inner_cv=10, max_run_time=1500, n_iter=100)

# Fitting...
opt.fit(X, y)

# Print best parameter setting and corresponding score
print(opt.best_params_, opt.best_score_)
```

### Using Optimus with OpenML
```python
from sklearn.preprocessing import StandardScaler, RobustScaler
from optimus_ml import ModelOptimizer
from optimus_ml import converter
from sklearn.svm import SVC
import numpy as np
import openml

# Setup classifier
clf = SVC(probability=True, random_state=3)

# Setup parameter grid, including some preprocessors with the special @preprocessor key
param_grid = {
    "C": np.logspace(-10, 10, num=21, base=2).tolist(),
    "gamma": np.logspace(-10, 0, num=11, base=2).tolist(),
    "kernel": ["linear", "poly", "rbf"],
    "@preprocessor": [
        StandardScaler(),
        RobustScaler()
    ]
}

# Convert the param_grid to a JSON serializable format for compatibility with OpenML
encoded_grid = converter.grid_to_json(param_grid)

# Setup Model Optimizer. The Model Optimizer knows how to decode an encoded grid.
opt = ModelOptimizer(estimator=clf, encoded_params=encoded_grid, inner_cv=10, max_run_time=1500, n_iter=10)

# Choose a task to run on
task = openml.tasks.get_task(12)

# Run the model on the task
run = openml.runs.run_model_on_task(task, opt)

# Publish the task to OpenML
run.publish()
```

### Reproducing runs from OpenML
```python
from optimus_ml.extra import oml
from openml import runs

# A (temporary) wrapper around the original `initialize_model_from_trace()` to 
# handle extra functionality, such as decoding dictionaries.
print(oml.initialize_model_from_trace(5709844, 0, 0, 0))

# The default OpenML function.
print(runs.initialize_model_from_run(5709844))
```

## Method comparison
Below you'll see a comparison of a few different methods that are available. 
The graph displays average rank over time (in seconds), where lower ranks are better.
We use different classifiers for estimating the expected improvement (EI) and running time.
The expected improvement is not calculated when doing a Randomized Search, and the running time is only calculated in the 
`EI/s` methods. "LS" means that Local Search was enabled.

The maximization time is defined as the time to find a setting with the highest EI. The evaluation time 
is defined as the time it takes to evaluate a setting.
 
Each method was executed three times on the following ten Openml tasks: [12](https://www.openml.org/t/12), [14](https://www.openml.org/t/14), [16](https://www.openml.org/t/16), [20](https://www.openml.org/t/20), [22](https://www.openml.org/t/22), [28](https://www.openml.org/t/28), [32](https://www.openml.org/t/32), [41](https://www.openml.org/t/41), [45](https://www.openml.org/t/45) and [58](https://www.openml.org/t/58), using a Random Forest Classifier. For a better comparison of the different methods, each method is pre-seeded with the same knowledge about five hyper parameter settings.
<img src="http://jeroenvanhoof.nl/optimus/ranking-v4.png"/>
<img src="http://jeroenvanhoof.nl/optimus/speed-v4.png"/>
<img src="http://jeroenvanhoof.nl/optimus/eval-time-v4.png"/>
<img src="http://jeroenvanhoof.nl/optimus/max-time-v4.png"/>