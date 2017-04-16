import numpy as np
from abc import abstractmethod, ABCMeta


class ParamDistribution:
    def __init__(self, param_distribution):
        self.params = param_distribution

    def get_bounds(self):
        result = []
        for key, transformer in self.params:
            result.append(transformer.get_bounds())
        return result

    def get_random(self):
        result = []
        for key, transformer in self.params:
            result.append(transformer.get_random())
        return result

    def realistic(self, vector):
        result = []
        for index, param in enumerate(self.params):
            transformer = self.params[param]  # type: Transformer
            result[index] = transformer.get_used_part_of_value(vector[index])
        return result

    def transform_to_params(self, vector):
        result = {}
        for index, param in enumerate(self.params):
            transformer = self.params[param]  # type: Transformer
            result[param] = transformer.transform_to_param(vector[index])
        return result

    def transform_to_values(self, parameters):
        result = []
        for index, param in enumerate(parameters):
            transformer = self.params[param]  # type: Transformer
            result[index] = transformer.transform_to_value(parameters[param])
        return result


class Transformer(metaclass=ABCMeta):
    def get_bounds(self):
        raise NotImplementedError()

    def get_random(self):
        raise NotImplementedError()

    def get_used_part_of_value(self, value):
        raise NotImplementedError()

    def transform_to_param(self, value):
        raise NotImplementedError()

    def transform_to_value(self, value):
        raise NotImplementedError()


class Continuous(Transformer):
    def __init__(self, bounds):
        self.bounds = bounds

    def get_bounds(self):
        return self.bounds

    def get_random(self):
        return np.random.uniform(self.bounds[0], self.bounds[1])

    def get_used_part_of_value(self, value):
        return value

    def transform_to_param(self, value):
        return value

    def transform_to_value(self, value):
        return value


class Choice(Transformer):
    def __init__(self, choices):
        # Take example set S = ["car", "bike", "airplane"].
        # transform(2.9) -> "airplane"
        # transform(0.9) -> "car"
        # len(S) = 3
        self.choices = choices
        self.bounds = (0, len(self.choices))

    def get_bounds(self):
        return self.bounds

    def transform_to_param(self, value):
        # Given an example set of length 3, value 2.9 would be transformed to
        # 2, but value 3 would be 3, which would be out of bounds. So we set
        # the upper bound to length - 1 using the min() function.
        index = min(int(value), len(self.choices) - 1)
        return self.choices[index]

    def transform_to_value(self, param):
        # "bike" --> 1
        return self.choices.index(param)

    def get_random(self):
        # Given S, this should return one of [0, 1, 2]
        return np.random.randint(0, len(self.choices))

    def get_used_part_of_value(self, value):
        return min(int(value), len(self.choices) - 1)


class LogScale(Transformer):
    def __init__(self, base, bounds, discrete_exponent=False, discrete_result=False):
        self.base = base
        self.bounds = bounds
        self.discrete_exponent = discrete_exponent
        self.discrete_result = discrete_result

        # We add 1 because we round down
        if discrete_exponent:
            self.disc = Discrete(bounds)
            # self.bounds = (bounds[0], bounds[1] + 1)

    def get_bounds(self):
        return self.bounds

    def transform_to_param(self, value):
        if self.discrete_exponent:
            return self.base ** self.disc.transform_to_param(value)

        elif self.discrete_result:
            return int(np.round(self.base ** value))

        return self.base ** value

    def transform_to_value(self, param):
        return param ** (1 / self.base)

    def get_random(self):
        if self.discrete_exponent:
            return self.disc.get_random()

        return np.random.uniform(self.bounds[0], self.bounds[1])

    def get_used_part_of_value(self, value):
        if self.discrete_exponent:
            return self.disc.get_used_part_of_value(value)
        else:
            return value


class Discrete(Transformer):
    def __init__(self, bounds):
        # We add 1 to the 'high' bound here, because we will round down.
        # Take example set S = [0, 1, ..., 8, 9]
        # transform(9.9) -> 9
        # transform(0.9) -> 0
        self.bounds = (bounds[0], bounds[1] + 1)

    def get_random(self):
        # Using example set S again, we now have bounds (0, 10). We want this
        # function to return one of [0, 1, ..., 8, 9]. Because the 'high'
        # parameter is exclusive, we can just use the normal low and high from
        # the bounds, i.e. (0, 10)
        return np.random.randint(self.bounds[0], self.bounds[1])

    def transform_to_param(self, value):
        # Given example set S, the extreme value 10 would be transformed to 10,
        # which would be out of the boundaries, so we set the upper bound to
        # 10 - 1 using the min() function. Due to
        return max(min(int(value), self.bounds[1] - 1), self.bounds[0])

    def transform_to_value(self, param):
        # 3 --> 3
        return param

    def get_bounds(self):
        return self.bounds

    def get_used_part_of_value(self, value):
        return self.transform_to_param(value)


class MultiplyPlus(Transformer):
    def __init__(self, n, k, bounds):
        self.n = n
        self.k = k
        self.disc = Discrete(bounds)

    def get_bounds(self):
        return self.disc.get_bounds()

    def get_random(self):
        return self.disc.get_random()

    def transform_to_param(self, value):
        disc_value = self.disc.transform_to_param(value)
        return self.n * disc_value + self.k

    def transform_to_value(self, param):
        return (param - self.k) / self.n

    def get_used_part_of_value(self, value):
        return self.disc.get_used_part_of_value(value)
