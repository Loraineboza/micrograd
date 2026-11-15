#!/usr/bin/env python3
"""Print a reproducible float64 central-difference check for X @ W + b."""

import numpy as np
from micrograd import Tensor


def loss_value(X, W, b, Y):
    return float(np.mean((X @ W + b - Y) ** 2))


def main():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, 3), dtype=np.float64)
    W = rng.standard_normal((3, 2), dtype=np.float64)
    b = np.array([0.5, -0.5], dtype=np.float64)
    Y = rng.standard_normal((8, 2), dtype=np.float64)

    xt = Tensor(X, requires_grad=False)
    wt = Tensor(W.copy(), name="W")
    bt = Tensor(b.copy(), name="b")
    loss = ((xt @ wt + bt - Tensor(Y, requires_grad=False)) ** 2).mean()
    loss.backward()

    eps = 1e-6
    numeric_w = np.zeros_like(W)
    numeric_b = np.zeros_like(b)

    for index in np.ndindex(W.shape):
        plus, minus = W.copy(), W.copy()
        plus[index] += eps
        minus[index] -= eps
        numeric_w[index] = (
            loss_value(X, plus, b, Y) - loss_value(X, minus, b, Y)
        ) / (2 * eps)

    for index in np.ndindex(b.shape):
        plus, minus = b.copy(), b.copy()
        plus[index] += eps
        minus[index] -= eps
        numeric_b[index] = (
            loss_value(X, W, plus, Y) - loss_value(X, W, minus, Y)
        ) / (2 * eps)

    max_w = float(np.max(np.abs(wt.grad - numeric_w)))
    max_b = float(np.max(np.abs(bt.grad - numeric_b)))
    tolerance = 1e-6 + 1e-4 * max(
        float(np.max(np.abs(numeric_w))),
        float(np.max(np.abs(numeric_b))),
    )

    print("dtype: float64")
    print(f"finite_difference_step: {eps:.1e}")
    print(f"tolerance: {tolerance:.12e}")
    print(f"max_abs_error_W: {max_w:.12e}")
    print(f"max_abs_error_b: {max_b:.12e}")
    print("analytic_grad_W:")
    print(wt.grad)
    print("numeric_grad_W:")
    print(numeric_w)
    print("analytic_grad_b:", bt.grad)
    print("numeric_grad_b:", numeric_b)

    if max(max_w, max_b) > tolerance:
        raise SystemExit("Gradient check FAILED")
    print("Gradient check PASSED")


if __name__ == "__main__":
    main()
