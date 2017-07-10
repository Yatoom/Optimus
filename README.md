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

## Method comparison
Below you'll see a comparison of a few different methods that are available. 
The graph displays average rank over time (in seconds), where lower ranks are better.
We use different classifiers for estimating the expected improvement (EI) and running time.
The expected improvement is not calculated when doing a Randomized Search, and the running time is only calculated in the 
`EI/s` methods. 

The maximization time is defined as the time to find a setting with the highest EI. The evaluation time 
is defined as the time it takes to evaluate a setting.
 
Each method was executed three times on the following ten Openml tasks: [12](https://www.openml.org/t/12), [14](https://www.openml.org/t/14), [16](https://www.openml.org/t/16), [20](https://www.openml.org/t/20), [22](https://www.openml.org/t/22), [28](https://www.openml.org/t/28), [32](https://www.openml.org/t/32), [41](https://www.openml.org/t/41), [45](https://www.openml.org/t/45) and [58](https://www.openml.org/t/58), using a Random Forest Classifier. For a better comparison of the different methods, each method is pre-seeded with the same knowledge about five hyper parameter settings.
<img src="http://jeroenvanhoof.nl/optimus/ranking.png"/>
<img src="http://jeroenvanhoof.nl/optimus/speed.png"/>
<img src="http://jeroenvanhoof.nl/optimus/evaluation time.png"/>
<img src="http://jeroenvanhoof.nl/optimus/maximization time.png"/>