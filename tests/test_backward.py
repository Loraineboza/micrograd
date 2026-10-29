import numpy as np

import pytest

from micrograd import Tensor


def test_single_node_reuse_accumulates_grad():
    x = Tensor(np.array([2.0], dtype=np.float64), name="x")
    loss = (x * x + x).sum()
    loss.backward()
    assert np.allclose(x.grad, np.array([5.0], dtype=np.float64))

    x.zero_grad()
    loss = (x * x + x).sum()
    loss.backward()
    assert np.allclose(x.grad, np.array([5.0], dtype=np.float64))


def test_merging_branches_accumulates_grad():
    x = Tensor(np.array([2.0], dtype=np.float64), name="x")
    loss = (x * x + x * x).sum()
    loss.backward()
    assert np.allclose(x.grad, np.array([8.0], dtype=np.float64))


def test_broadcast_bias_grad_is_reduced_to_original_shape():
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64), name="x")
    b = Tensor(np.array([10.0, 20.0], dtype=np.float64), name="b")
    loss = (x + b).sum()
    loss.backward()
    assert np.allclose(b.grad, np.array([2.0, 2.0], dtype=np.float64))
    assert np.allclose(x.grad, np.ones_like(x.data, dtype=np.float64))


def test_repeat_backward_has_explicit_accumulation_semantics():
    x = Tensor(np.array([2.0], dtype=np.float64), name="x")
    loss = (x * x).sum()
    loss.backward()
    first_grad = x.grad.copy()
    loss.backward()
    assert np.allclose(x.grad, first_grad + np.array([4.0], dtype=np.float64))

    x.zero_grad()
    loss = (x * x).sum()
    loss.backward()
    assert np.allclose(x.grad, np.array([4.0], dtype=np.float64))


def test_unsupported_matmul_shapes_raise_clear_error():
    x = Tensor(np.ones((2, 2), dtype=np.float64), requires_grad=False)
    y = Tensor(np.ones((2, 2, 2), dtype=np.float64), requires_grad=False)
    with pytest.raises(ValueError, match="supports only 2D arrays"):
        _ = x @ y
