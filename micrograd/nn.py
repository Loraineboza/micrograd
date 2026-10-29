import numpy as np

from .autograd import Tensor


class Linear:
    def __init__(self, in_features, out_features, bias=True, seed=None):
        rng = np.random.default_rng(seed)
        limit = np.sqrt(2.0 / in_features)
        self.weight = Tensor(rng.normal(0.0, limit, size=(in_features, out_features)), name="weight")
        self.bias = Tensor(np.zeros(out_features), name="bias") if bias else None

    def __call__(self, x):
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class Sequential:
    def __init__(self, *layers):
        self.layers = list(layers)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MLP:
    def __init__(self, features, activation="relu"):
        self.activation_name = activation
        self.layers = []
        for in_f, out_f in zip(features[:-1], features[1:]):
            self.layers.append(Linear(in_f, out_f))

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            if self.activation_name == "relu":
                x = x.relu()
            elif self.activation_name == "tanh":
                x = x.tanh()
            elif self.activation_name == "sigmoid":
                x = x.sigmoid()
        if self.layers:
            x = self.layers[-1](x)
        return x


__all__ = ["Linear", "Sequential", "MLP"]
