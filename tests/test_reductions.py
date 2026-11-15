import numpy as np

from micrograd import Tensor


def test_sum_axis_one_keepdims_forward_and_backward_with_nonuniform_upstream_grad():
    x_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    row_weights = np.array([[2.0], [3.0]], dtype=np.float64)

    x = Tensor(x_data.copy())
    reduced = x.sum(axis=1, keepdims=True)

    assert reduced.data.shape == (2, 1)
    np.testing.assert_array_equal(reduced.data, np.array([[6.0], [15.0]]))

    loss = (reduced * row_weights).sum()
    loss.backward()

    np.testing.assert_array_equal(
        x.grad,
        np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], dtype=np.float64),
    )


def test_mean_axis_one_keepdims_backward():
    x_data = np.arange(1.0, 7.0, dtype=np.float64).reshape(2, 3)
    row_weights = np.array([[2.0], [4.0]], dtype=np.float64)
    x = Tensor(x_data.copy())

    loss = (x.mean(axis=1, keepdims=True) * row_weights).sum()
    loss.backward()

    np.testing.assert_allclose(
        x.grad,
        np.array([[2.0 / 3.0] * 3, [4.0 / 3.0] * 3], dtype=np.float64),
        rtol=0,
        atol=1e-12,
    )


def test_zero_grad_resets_existing_graph_and_all_leaf_gradients():
    x = Tensor(np.array([2.0], dtype=np.float64))
    intermediate = x * x
    loss = (intermediate + x).sum()

    loss.backward()
    np.testing.assert_array_equal(x.grad, np.array([5.0]))

    loss.zero_grad()
    np.testing.assert_array_equal(x.grad, np.array([0.0]))
    np.testing.assert_array_equal(loss.grad, np.array(0.0))

    loss.backward()
    np.testing.assert_array_equal(x.grad, np.array([5.0]))
