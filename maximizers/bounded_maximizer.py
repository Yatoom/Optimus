import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor

from optimus.converter import Converter
from vault.transformers import ParamDistribution


class BoundedMaximizer:
    def __init__(self, gp, param_distribution):
        self.gp = gp  # type: GaussianProcessRegressor
        self.param_distribution = param_distribution  # type: ParamDistribution
        self.bounds = self.param_distribution.get_bounds()

    def maximize(self, n_restarts, validated_params, validated_scores, current_best_score):
        validated_losses = - validated_scores
        best_value = np.inf
        best_x = None
        # validated_vectors = self.param_distribution.transform(validated_params)

        self.gp.fit(self.param_distribution.transform_to_values(validated_params), validated_losses)

        for _ in range(0, n_restarts):
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

        return best_x, -self.realize(validated_params, validated_losses, best_x)

    def realize(self, validated_params, validated_losses, point):
        params, scores = Converter.drop_zero_scores(validated_params, validated_losses)
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
