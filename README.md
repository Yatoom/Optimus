<p align="center">
    <img src="http://jeroenvanhoof.nl/optimus2.svg" width="50%"/>
</p>

# Optimus
An automated machine learning tool.

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