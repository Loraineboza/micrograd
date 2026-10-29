#!/usr/bin/env python3
"""Executable example for the required X @ W + b + MSE / backprop workflow."""

import numpy as np

from micrograd import Tensor


def main():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, 3), dtype=np.float64)
    W = rng.standard_normal((3, 2), dtype=np.float64)
    b = np.array([0.5, -0.5], dtype=np.float64)
    Y = rng.standard_normal((8, 2), dtype=np.float64)

    X_t = Tensor(X, requires_grad=False, name="X")
    W_t = Tensor(W, requires_grad=True, name="W")
    b_t = Tensor(b, requires_grad=True, name="b")
    Y_t = Tensor(Y, requires_grad=False, name="Y")

    P = X_t @ W_t + b_t
    L = ((P - Y_t) ** 2).mean()

    L.backward()

    print("X shape:", X.shape)
    print("W shape:", W.shape)
    print("b shape:", b.shape)
    print("Y shape:", Y.shape)
    print("P shape:", P.data.shape)
    print("loss:", float(L.data))
    print("grad W:\n", W_t.grad)
    print("grad b:", b_t.grad)


if __name__ == "__main__":
    main()
