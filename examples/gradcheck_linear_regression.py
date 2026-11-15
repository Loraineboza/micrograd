
import numpy as np

from micrograd import Tensor


def numpy_loss(X, W, b, Y):
    return float(np.mean((X @ W + b - Y) ** 2))


def numerical_gradient(X, W, b, Y, eps):
    grad_W = np.zeros_like(W)
    grad_b = np.zeros_like(b)

    for index in np.ndindex(W.shape):
        plus = W.copy()
        minus = W.copy()
        plus[index] += eps
        minus[index] -= eps

        grad_W[index] = (
            numpy_loss(X, plus, b, Y)
            - numpy_loss(X, minus, b, Y)
        ) / (2 * eps)

    for index in np.ndindex(b.shape):
        plus = b.copy()
        minus = b.copy()
        plus[index] += eps
        minus[index] -= eps

        grad_b[index] = (
            numpy_loss(X, W, plus, Y)
            - numpy_loss(X, W, minus, Y)
        ) / (2 * eps)

    return grad_W, grad_b


def main():
    rng = np.random.default_rng(0)

    X = rng.standard_normal((8, 3), dtype=np.float64)
    W = rng.standard_normal((3, 2), dtype=np.float64)
    b = rng.standard_normal((2,), dtype=np.float64)
    Y = rng.standard_normal((8, 2), dtype=np.float64)

    X_t = Tensor(X)
    W_t = Tensor(W.copy())
    b_t = Tensor(b.copy())
    Y_t = Tensor(Y)

    loss = ((X_t @ W_t + b_t - Y_t) ** 2).mean()
    loss.backward()

    eps = 1e-6
    atol = 1e-6
    rtol = 1e-4

    numeric_W, numeric_b = numerical_gradient(X, W, b, Y, eps)

    error_W = np.max(np.abs(W_t.grad - numeric_W))
    error_b = np.max(np.abs(b_t.grad - numeric_b))

    print("dtype: float64")
    print(f"finite difference step: {eps}")
    print(f"atol: {atol}, rtol: {rtol}")
    print(f"loss: {loss.data}")
    print(f"max absolute error W: {error_W}")
    print(f"max absolute error b: {error_b}")

    np.testing.assert_allclose(
        W_t.grad, numeric_W, atol=atol, rtol=rtol
    )
    np.testing.assert_allclose(
        b_t.grad, numeric_b, atol=atol, rtol=rtol
    )

    print("gradient check: passed")


if __name__ == "__main__":
    main()
