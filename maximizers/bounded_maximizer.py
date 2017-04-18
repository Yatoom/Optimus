import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor

from optimus.converter import Converter
from vault.transformers import ParamDistribution


class BoundedMaximizer:
    def __init__(self, gp, param_distribution, n_restarts=10):
        self.gp = gp  # type: GaussianProcessRegressor
        self.param_distribution = ParamDistribution(param_distribution)
        self.bounds = self.param_distribution.get_bounds()
        self.n_restarts = n_restarts

    def maximize(self, validated_params, validated_scores, current_best_score):
        if len(validated_scores) == 0:
            random_x = self.param_distribution.get_random()
            params = self.param_distribution.transform_to_params(random_x)
            return params, 0

        validated_losses = (- np.array(validated_scores)).tolist()
        best_value = np.inf
        best_x = None

        self.gp.fit(self.param_distribution.transform_to_values(validated_params), validated_losses)

        for _ in range(0, self.n_restarts):
            point = self.param_distribution.get_random()
            res = minimize(
                fun=self.get_ei,
                args=(current_best_score,),
                x0=point,
                bounds=self.bounds,
                method="L-BFGS-B"
            )

            if res.fun < best_value:
                best_value = res.fun
                best_x = res.x

        best_params = self.param_distribution.transform_to_params(best_x)
        best_value = - self.realize(validated_params, validated_losses, best_x, best_value)

        return best_params, best_value

    def realize(self, validated_params, validated_losses, point, original):
        params, scores = Converter.remove_timeouts(validated_params, validated_losses)

        if len(scores) == 0:
            return original

        values = self.param_distribution.transform_to_values(params)
        self.gp.fit(values, scores)
        return self.gp.predict(self.param_distribution.realistic(point))

    def get_ei(self, point, current_best_score):
        point = np.array(point).reshape(1, -1)
        mu, sigma = self.gp.predict(point, return_std=True)
        best_score = current_best_score
        mu = mu[0]
        sigma = sigma[0]

        # We want our mu to be lower than the loss optimum
        diff = best_score - mu - 0.01
        if sigma == 0:
            return 0
        Z = diff / sigma
        ei = diff * norm.cdf(Z) + sigma * norm.pdf(Z)
        return -ei
