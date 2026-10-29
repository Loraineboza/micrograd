import numpy as np

from micrograd import Tensor


def _loss_and_grad_linear(X, W, b, Y):
    X_t = Tensor(X.copy(), requires_grad=False, name="X")
    W_t = Tensor(W.copy(), requires_grad=True, name="W")
    b_t = Tensor(b.copy(), requires_grad=True, name="b")
    Y_t = Tensor(Y.copy(), requires_grad=False, name="Y")
    loss = ((X_t @ W_t + b_t - Y_t) ** 2).mean()
    loss.backward()
    return float(loss.data), W_t.grad.copy(), b_t.grad.copy()


def _assert_gradcheck_scalar(analytic, numeric, *, label):
    analytic = np.asarray(analytic, dtype=np.float64)
    numeric = np.asarray(numeric, dtype=np.float64)
    max_err = np.max(np.abs(analytic - numeric))
    ref_scale = np.max(np.abs(numeric)) if numeric.size else 0.0
    tol = 1e-6 + 1e-4 * ref_scale
    assert max_err <= tol, (
        f"{label}: analytic={analytic}, numeric={numeric}, max_err={max_err}, tol={tol}"
    )


def test_gradcheck_for_linear_regression_example():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, 3), dtype=np.float64)
    W = rng.standard_normal((3, 2), dtype=np.float64)
    b = rng.standard_normal((2,), dtype=np.float64)
    Y = rng.standard_normal((8, 2), dtype=np.float64)

    loss0, grad_W, grad_b = _loss_and_grad_linear(X, W, b, Y)
    assert np.isfinite(loss0)
    assert grad_W.shape == W.shape
    assert grad_b.shape == b.shape

    eps = 1e-6

    for idx in np.ndindex(W.shape):
        W_plus = W.copy()
        W_minus = W.copy()
        W_plus[idx] += eps
        W_minus[idx] -= eps
        loss_plus, _, _ = _loss_and_grad_linear(X, W_plus, b, Y)
        loss_minus, _, _ = _loss_and_grad_linear(X, W_minus, b, Y)
        numeric = (loss_plus - loss_minus) / (2 * eps)
        analytic = grad_W[idx]
        _assert_gradcheck_scalar(analytic, numeric, label=f"W[{idx}]")

    for idx in np.ndindex(b.shape):
        b_plus = b.copy()
        b_minus = b.copy()
        b_plus[idx] += eps
        b_minus[idx] -= eps
        loss_plus, _, _ = _loss_and_grad_linear(X, W, b_plus, Y)
        loss_minus, _, _ = _loss_and_grad_linear(X, W, b_minus, Y)
        numeric = (loss_plus - loss_minus) / (2 * eps)
        analytic = grad_b[idx]
        _assert_gradcheck_scalar(analytic, numeric, label=f"b[{idx}]")


def test_gradcheck_for_sum_axis():
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    x_t = Tensor(x.copy(), requires_grad=True, name="x")
    loss = x_t.sum(axis=1).sum()
    loss.backward()
    analytic = x_t.grad.copy()

    eps = 1e-6
    grad_numeric = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[idx] += eps
        x_minus[idx] -= eps
        f_plus = np.sum(np.sum(x_plus, axis=1))
        f_minus = np.sum(np.sum(x_minus, axis=1))
        grad_numeric[idx] = (f_plus - f_minus) / (2 * eps)

    _assert_gradcheck_scalar(analytic, grad_numeric, label="sum(axis=1)")


def test_gradcheck_for_mean_axis():
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    x_t = Tensor(x.copy(), requires_grad=True, name="x")
    loss = x_t.mean(axis=1).sum()
    loss.backward()
    analytic = x_t.grad.copy()

    eps = 1e-6
    grad_numeric = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[idx] += eps
        x_minus[idx] -= eps
        f_plus = np.mean(x_plus, axis=1).sum()
        f_minus = np.mean(x_minus, axis=1).sum()
        grad_numeric[idx] = (f_plus - f_minus) / (2 * eps)

    _assert_gradcheck_scalar(analytic, grad_numeric, label="mean(axis=1)")


def test_gradcheck_for_matmul():
    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    B = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)

    A_t = Tensor(A.copy(), requires_grad=True, name="A")
    B_t = Tensor(B.copy(), requires_grad=True, name="B")
    loss = (A_t @ B_t).sum()
    loss.backward()
    analytic_A = A_t.grad.copy()
    analytic_B = B_t.grad.copy()

    eps = 1e-6
    grad_numeric_A = np.zeros_like(A)
    grad_numeric_B = np.zeros_like(B)

    for idx in np.ndindex(A.shape):
        A_plus = A.copy(); A_minus = A.copy()
        A_plus[idx] += eps; A_minus[idx] -= eps
        grad_numeric_A[idx] = ((A_plus @ B).sum() - (A_minus @ B).sum()) / (2 * eps)

    for idx in np.ndindex(B.shape):
        B_plus = B.copy(); B_minus = B.copy()
        B_plus[idx] += eps; B_minus[idx] -= eps
        grad_numeric_B[idx] = ((A @ B_plus).sum() - (A @ B_minus).sum()) / (2 * eps)

    _assert_gradcheck_scalar(analytic_A, grad_numeric_A, label="matmul A gradient")
    _assert_gradcheck_scalar(analytic_B, grad_numeric_B, label="matmul B gradient")


def test_gradcheck_for_broadcast_bias_after_sum():
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    b = np.array([10.0, 20.0], dtype=np.float64)

    x_t = Tensor(x.copy(), requires_grad=True, name="x")
    b_t = Tensor(b.copy(), requires_grad=True, name="b")
    loss = (x_t + b_t).sum()
    loss.backward()

    analytic_b = b_t.grad.copy()
    eps = 1e-6
    numeric_b = np.zeros_like(b)
    for idx in np.ndindex(b.shape):
        b_plus = b.copy(); b_minus = b.copy()
        b_plus[idx] += eps; b_minus[idx] -= eps
        numeric_b[idx] = ((x + b_plus).sum() - (x + b_minus).sum()) / (2 * eps)

    _assert_gradcheck_scalar(analytic_b, numeric_b, label="broadcast bias grad")


def test_gradcheck_for_reused_node_and_branching():
    x = np.array([2.0], dtype=np.float64)
    x_t = Tensor(x.copy(), requires_grad=True, name="x")
    loss = (x_t * x_t + x_t).sum()
    loss.backward()
    analytic = x_t.grad.copy()

    eps = 1e-6
    x_plus = x.copy(); x_minus = x.copy()
    x_plus[0] += eps; x_minus[0] -= eps
    numeric = ((x_plus[0] * x_plus[0] + x_plus[0]) - (x_minus[0] * x_minus[0] + x_minus[0])) / (2 * eps)
    _assert_gradcheck_scalar(analytic, np.array([numeric]), label="single-node reuse")

    x2 = np.array([2.0], dtype=np.float64)
    x2_t = Tensor(x2.copy(), requires_grad=True, name="x2")
    loss2 = (x2_t * x2_t + x2_t * x2_t).sum()
    loss2.backward()
    analytic2 = x2_t.grad.copy()
    numeric2 = ( (2.0 + eps) ** 2 + (2.0 + eps) ** 2 - ((2.0 - eps) ** 2 + (2.0 - eps) ** 2) ) / (2 * eps)
    _assert_gradcheck_scalar(analytic2, np.array([numeric2]), label="branching")


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
