import numpy as np

from micrograd import Tensor


def test_repeated_backward_accumulates_leaf_gradients_three_times():
    x = Tensor([2.0], name="x")
    loss = (x * x).sum()

    observed = []
    for _ in range(3):
        loss.backward()
        observed.append(float(x.grad[0]))

    assert observed == [4.0, 8.0, 12.0]

    x.zero_grad()
    loss.backward()
    np.testing.assert_array_equal(x.grad, np.array([4.0]))


def test_repeated_backward_on_converging_branches():
    x = Tensor([2.0], name="x")
    shared = x * x
    loss = (shared + shared).sum()

    loss.backward()
    np.testing.assert_array_equal(x.grad, np.array([8.0]))
    loss.backward()
    np.testing.assert_array_equal(x.grad, np.array([16.0]))


def test_repeated_backward_does_not_reuse_intermediate_gradients():
    x = Tensor([2.0], name="x")
    intermediate = x * x
    loss = intermediate.sum()

    loss.backward()
    assert float(intermediate.grad[0]) == 1.0
    loss.backward()
    assert float(intermediate.grad[0]) == 1.0
    np.testing.assert_array_equal(x.grad, np.array([8.0]))
