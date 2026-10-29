import numpy as np

import pytest

from micrograd import Tensor


def _loss_value(W, b, X, Y):
    W_t = Tensor(W.copy(), requires_grad=True, name="W")
    b_t = Tensor(b.copy(), requires_grad=True, name="b")
    X_t = Tensor(X, requires_grad=False, name="X")
    Y_t = Tensor(Y, requires_grad=False, name="Y")
    loss = ((X_t @ W_t + b_t - Y_t) ** 2).mean()
    loss.backward()
    return float(loss.data), W_t.grad.copy(), b_t.grad.copy()


def test_gradcheck_for_linear_regression_example():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, 3), dtype=np.float64)
    W = rng.standard_normal((3, 2), dtype=np.float64)
    b = rng.standard_normal((2,), dtype=np.float64)
    Y = rng.standard_normal((8, 2), dtype=np.float64)

    loss0, grad_W, grad_b = _loss_value(W, b, X, Y)
    assert np.isfinite(loss0)
    assert grad_W.shape == W.shape
    assert grad_b.shape == b.shape

    eps = 1e-6
    tol = 1e-6 + 1e-4

    for idx in np.ndindex(W.shape):
        W_plus = W.copy()
        W_minus = W.copy()
        W_plus[idx] += eps
        W_minus[idx] -= eps
        loss_plus, _, _ = _loss_value(W_plus, b, X, Y)
        loss_minus, _, _ = _loss_value(W_minus, b, X, Y)
        numeric = (loss_plus - loss_minus) / (2 * eps)
        max_err = abs(grad_W[idx] - numeric)
        assert max_err <= tol * max(1.0, abs(numeric)), (
            f"W gradcheck failed at {idx}: analytic={grad_W[idx]}, numeric={numeric}, error={max_err}"
        )

    for idx in np.ndindex(b.shape):
        b_plus = b.copy()
        b_minus = b.copy()
        b_plus[idx] += eps
        b_minus[idx] -= eps
        _, _, grad_b_plus = _loss_value(W, b_plus, X, Y)[1:]
        _, _, grad_b_minus = _loss_value(W, b_minus, X, Y)[1:]
        _, _, grad_b = _loss_value(W, b, X, Y)[1:]
        numeric = (grad_b_plus[idx] - grad_b_minus[idx]) / (2 * eps)
        analytic = grad_b[idx]
        max_err = abs(analytic - numeric)
        assert max_err <= tol * max(1.0, abs(numeric)), (
            f"b gradcheck failed at {idx}: analytic={analytic}, numeric={numeric}, error={max_err}"
        )


def test_gradcheck_for_reused_node_and_branching():
    x = Tensor(np.array([2.0], dtype=np.float64), requires_grad=True, name="x")
    loss = (x * x + x).sum()
    loss.backward()
    analytic = x.grad.copy()

    eps = 1e-6
    x_plus = x.data.copy() + eps
    x_minus = x.data.copy() - eps
    f_plus = ((x_plus * x_plus + x_plus).sum())
    f_minus = ((x_minus * x_minus + x_minus).sum())
    numeric = (f_plus - f_minus) / (2 * eps)
    assert abs(analytic[0] - numeric) <= 1e-6 + 1e-4 * abs(numeric)

    x2 = Tensor(np.array([2.0], dtype=np.float64), requires_grad=True, name="x2")
    loss2 = (x2 * x2 + x2 * x2).sum()
    loss2.backward()
    analytic2 = x2.grad.copy()
    numeric2 = 8.0
    assert abs(analytic2[0] - numeric2) <= 1e-6 + 1e-4 * abs(numeric2)


def test_gradcheck_for_broadcast_bias_after_sum():
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64), name="x")
    b = Tensor(np.array([10.0, 20.0], dtype=np.float64), name="b")
    loss = (x + b).sum()
    loss.backward()

    numeric = np.array([2.0, 2.0], dtype=np.float64)
    assert np.allclose(b.grad, numeric)


def test_repeat_backward_rule_is_documented_and_tested():
    x = Tensor(np.array([2.0], dtype=np.float64), name="x")
    loss = (x * x).sum()
    loss.backward()
    first = x.grad.copy()
    loss.backward()
    assert np.allclose(x.grad, first + np.array([4.0], dtype=np.float64))

    x.zero_grad()
    loss = (x * x).sum()
    loss.backward()
    assert np.allclose(x.grad, np.array([4.0], dtype=np.float64))
