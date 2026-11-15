import numpy as np

from micrograd import Tensor


def test_sum_axis_keepdims_all_requested_shapes():
    data = np.arange(1.0, 7.0, dtype=np.float64).reshape(2, 3)

    cases = [
        (0, True, (1, 3)),
        (0, False, (3,)),
        (1, True, (2, 1)),
        (1, False, (2,)),
        (-1, True, (2, 1)),
        ((0, 1), True, (1, 1)),
        ((0, 1), False, ()),
    ]

    for axis, keepdims, expected_shape in cases:
        x = Tensor(data.copy(), name="x")
        reduced = x.sum(axis=axis, keepdims=keepdims)
        assert reduced.data.shape == expected_shape
        reduced.sum().backward()
        np.testing.assert_allclose(x.grad, np.ones_like(data))


def test_nested_sum_keepdims_regression_from_review():
    x = Tensor(np.arange(1.0, 7.0).reshape(2, 3), name="x")
    x.sum(axis=1, keepdims=True).sum().backward()
    np.testing.assert_array_equal(x.grad, np.ones((2, 3), dtype=np.float64))


def test_sum_keepdims_gradient_after_repeated_backward_accumulates_leaves():
    x = Tensor(np.arange(1.0, 7.0).reshape(2, 3), name="x")
    loss = x.sum(axis=1, keepdims=True).sum()
    loss.backward()
    loss.backward()
    np.testing.assert_array_equal(x.grad, np.full((2, 3), 2.0))
