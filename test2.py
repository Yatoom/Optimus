from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline

from extra.dual_imputer import DualImputer
from optimus.builder import Builder
from vault import models, data
import openml

from prime.search import Optimizer

task = openml.tasks.get_task(2071)

dataset = task.get_dataset()
X, y, categorical, names = dataset.get_data(
    target=dataset.default_target_attribute,
    return_categorical_indicator=True,
    return_attribute_names=True
)

params = {'max_depth': 2, 'learning_rate': 0.19306977288832497, "@preprocessor": DualImputer(categorical=categorical)}
base_model = GradientBoostingClassifier(random_state=3, n_estimators=512)
pipeline = Builder.build_pipeline(GradientBoostingClassifier(random_state=3, n_estimators=512), params)
print(pipeline)
run = openml.runs.run_task(task, pipeline)
run.publish()


