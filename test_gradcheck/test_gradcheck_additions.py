"""Tests for scalar autograd operations, broadcasting and gradcheck.

Assumes the project exposes `Tensor` from `micrograd.autograd` with:
- Tensor(data), .data, .grad, .backward()
- arithmetic +, *, @, sum(axis=...), mean(axis=...)
If the actual public API differs, adapt the import/API in this file rather than changing
the user's implementation blindly.
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    from micrograd.autograd import Tensor
except ImportError as exc:  # Keep failure explicit if the repo API is different.
    raise ImportError(
        "Could not import Tensor from micrograd.autograd. Adjust the import to the "
        "actual public API in the repository before running these tests."
    ) from exc


def as_array(value):
    """Read a Tensor's numerical value without assuming a specific storage backend."""
    raw = value.data if hasattr(value, "data") else value
    return np.asarray(raw, dtype=np.float64)


def as_grad(value):
    grad = value.grad
    return np.asarray(grad, dtype=np.float64)


def central_difference(function, array, eps=1e-6):
    """Numerically differentiate scalar function with respect to every array element."""
    x = np.asarray(array, dtype=np.float64).copy()
    result = np.zeros_like(x, dtype=np.float64)
    for index in np.ndindex(x.shape):
        original = x[index]
        x[index] = original + eps
        plus = float(function(x))
        x[index] = original - eps
        minus = float(function(x))
        x[index] = original
        result[index] = (plus - minus) / (2.0 * eps)
    return result


def assert_gradclose(analytic, numeric, atol=1e-6, rtol=1e-4):
    analytic = np.asarray(analytic, dtype=np.float64)
    numeric = np.asarray(numeric, dtype=np.float64)
    assert analytic.shape == numeric.shape
    error = np.abs(analytic - numeric)
    allowed = atol + rtol * np.abs(numeric)
    assert np.all(error <= allowed), (
        f"gradcheck failed: max_abs_error={error.max():.8g}; "
        f"allowed_max={allowed.max():.8g}; analytic={analytic}; numeric={numeric}"
    )


def scalar_loss(value):
    """Return a Python float from a scalar Tensor or numeric scalar."""
    array = as_array(value)
    assert array.size == 1, f"Expected scalar loss, got shape {array.shape}"
    return float(array.reshape(()))


def test_add_and_multiply_gradients():
    a = Tensor(np.array([1.2, -0.7], dtype=np.float64))
    b = Tensor(np.array([0.3, 2.0], dtype=np.float64))
    loss = (a * b + a).sum()
    loss.backward()
    np.testing.assert_allclose(as_grad(a), np.array([1.3, 3.0]), rtol=0, atol=1e-12)
    np.testing.assert_allclose(as_grad(b), np.array([1.2, -0.7]), rtol=0, atol=1e-12)


def test_matmul_gradient_shapes_and_values():
    rng = np.random.default_rng(17)
    x_data = rng.normal(size=(4, 3)).astype(np.float64)
    w_data = rng.normal(size=(3, 2)).astype(np.float64)
    x = Tensor(x_data.copy())
    w = Tensor(w_data.copy())
    loss = (x @ w).sum()
    loss.backward()

    np.testing.assert_allclose(as_grad(x), np.ones((4, 2)) @ w_data.T, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(as_grad(w), x_data.T @ np.ones((4, 2)), rtol=1e-10, atol=1e-10)


def test_bias_broadcast_gradient_sums_over_batch_axis():
    rng = np.random.default_rng(23)
    x_data = rng.normal(size=(8, 3)).astype(np.float64)
    w_data = rng.normal(size=(3, 2)).astype(np.float64)
    b_data = rng.normal(size=(2,)).astype(np.float64)
    y_data = rng.normal(size=(8, 2)).astype(np.float64)

    x = Tensor(x_data)
    w = Tensor(w_data)
    b = Tensor(b_data)
    y = Tensor(y_data)
    prediction = x @ w + b
    loss = ((prediction - y) * (prediction - y)).mean()
    loss.backward()

    numeric_b = central_difference(
        lambda b_value: np.mean((x_data @ w_data + b_value - y_data) ** 2),
        b_data,
    )
    assert_gradclose(as_grad(b), numeric_b)


def test_weight_gradient_finite_difference():
    rng = np.random.default_rng(31)
    x_data = rng.normal(size=(5, 3)).astype(np.float64)
    w_data = rng.normal(size=(3, 2)).astype(np.float64)
    b_data = rng.normal(size=(2,)).astype(np.float64)
    y_data = rng.normal(size=(5, 2)).astype(np.float64)

    x = Tensor(x_data)
    w = Tensor(w_data.copy())
    b = Tensor(b_data)
    y = Tensor(y_data)
    prediction = x @ w + b
    loss = ((prediction - y) * (prediction - y)).mean()
    loss.backward()

    numeric_w = central_difference(
        lambda w_value: np.mean((x_data @ w_value + b_data - y_data) ** 2),
        w_data,
    )
    assert_gradclose(as_grad(w), numeric_w)


def test_sum_of_x_squared_plus_x_gradcheck():
    x_data = np.array([-1.2, 0.4, 2.1], dtype=np.float64)
    x = Tensor(x_data.copy())
    loss = (x * x + x).sum()
    loss.backward()
    numeric = central_difference(lambda z: np.sum(z * z + z), x_data)
    assert_gradclose(as_grad(x), numeric)


def test_two_branches_converging_at_same_node_accumulate_gradients():
    x = Tensor(np.array([2.0, -3.0], dtype=np.float64))
    branch_a = x * x
    branch_b = x * 3.0
    loss = (branch_a + branch_b).sum()
    loss.backward()
    # d(x^2 + 3x)/dx = 2x + 3
    np.testing.assert_allclose(as_grad(x), np.array([7.0, -3.0]), rtol=0, atol=1e-12)


def test_repeated_backward_behavior_is_explicit_and_does_not_corrupt_graph():
    x = Tensor(np.array([2.0], dtype=np.float64))
    loss = (x * x).sum()
    loss.backward()
    first = as_grad(x).copy()
    loss.backward()
    second = as_grad(x).copy()

    # Different engines intentionally choose either accumulation or reset semantics.
    # This test accepts either behavior, but requires the second call to be finite and
    # to follow one consistent policy: unchanged (reset) or exactly doubled (accumulate).
    assert np.isfinite(second).all()
    assert np.allclose(second, first) or np.allclose(second, 2.0 * first), (
        f"Unexpected repeated-backward behavior: first={first}, second={second}"
    )


def test_gradcheck_reports_max_absolute_error():
    x_data = np.array([0.25, -0.5, 1.5], dtype=np.float64)
    x = Tensor(x_data.copy())
    loss = (x * x + x).sum()
    loss.backward()
    numeric = central_difference(lambda z: np.sum(z * z + z), x_data)
    error = np.max(np.abs(as_grad(x) - numeric))
    print(f"max absolute gradient error: {error:.12g}")
    assert error <= 1e-6 + 1e-4 * np.max(np.abs(numeric))
