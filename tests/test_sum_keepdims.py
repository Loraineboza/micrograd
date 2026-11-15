import numpy as np
import pytest

from micrograd import Tensor


@pytest.mark.parametrize("axis", [0, 1, -1, -2, (0, 1), (-2, -1)])
@pytest.mark.parametrize("keepdims", [True, False])
def test_sum_axis_keepdims_backward(axis, keepdims):
    data = np.arange(1.0, 7.0, dtype=np.float64).reshape(2, 3)
    x = Tensor(data.copy())

    loss = x.sum(axis=axis, keepdims=keepdims).sum()
    loss.backward()

    np.testing.assert_array_equal(x.grad, np.ones_like(data))


def test_sum_axis_one_keepdims_nested_sum_backward():
    data = np.arange(1.0, 7.0, dtype=np.float64).reshape(2, 3)
    x = Tensor(data.copy())

    loss = x.sum(axis=1, keepdims=True).sum()
    loss.backward()

    np.testing.assert_array_equal(
        x.grad,
        np.ones((2, 3), dtype=np.float64)
    )
