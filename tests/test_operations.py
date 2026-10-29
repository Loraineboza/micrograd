import numpy as np

from micrograd import Tensor


def test_elementwise_add_and_shape():
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=False)
    y = Tensor(np.array([[10.0, 20.0], [30.0, 40.0]]), requires_grad=False)
    z = x + y
    assert z.data.shape == (2, 2)
    assert np.allclose(z.data, np.array([[11.0, 22.0], [33.0, 44.0]]))


def test_matmul_and_sum():
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=False)
    w = Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]), requires_grad=False)
    y = x @ w
    assert y.data.shape == (2, 2)
    assert np.allclose(y.data, np.array([[19.0, 22.0], [43.0, 50.0]]))

    s = y.sum(axis=1)
    assert s.data.shape == (2,)
    assert np.allclose(s.data, np.array([41.0, 93.0]))


def test_broadcast_add():
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=False)
    b = Tensor(np.array([10.0, 20.0]), requires_grad=False)
    z = x + b
    assert z.data.shape == (2, 2)
    assert np.allclose(z.data, np.array([[11.0, 22.0], [13.0, 24.0]]))


def test_mean_and_axis_keepdims():
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=False)
    m = x.mean(axis=1, keepdims=True)
    assert m.data.shape == (2, 1)
    assert np.allclose(m.data, np.array([[1.5], [3.5]]))
