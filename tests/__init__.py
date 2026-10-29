import unittest

from micrograd.autograd import Tensor
from micrograd.losses import MSE, BinaryCrossEntropy
from micrograd.nn import Linear


class TestAutograd(unittest.TestCase):
    def test_addition_backward(self):
        x = Tensor(3.0, name="x")
        y = x * x + 2 * x + 1
        y.backward()
        self.assertAlmostEqual(x.grad, 8.0, places=6)

    def test_mse_backward(self):
        pred = Tensor([1.0, 2.0, 3.0], name="pred")
        target = Tensor([1.0, 0.0, 3.0], name="target")
        loss = MSE(pred, target)
        loss.backward()
        expected = 2.0 * (pred.data - target.data) / pred.data.size
        self.assertTrue((pred.grad == expected).all())

    def test_bce_backward(self):
        pred = Tensor([0.8, 0.2], name="pred")
        target = Tensor([1.0, 0.0], name="target")
        loss = BinaryCrossEntropy(pred, target)
        loss.backward()
        expected = -(target.data / pred.data - (1.0 - target.data) / (1.0 - pred.data)) / pred.data.size
        self.assertTrue(np.allclose(pred.grad, expected))

    def test_matrix_mul_backward(self):
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], name="x")
        w = Tensor([[1.0], [2.0]], name="w")
        out = x @ w
        loss = out.sum()
        loss.backward()
        self.assertEqual(x.grad.shape, x.data.shape)
        self.assertEqual(w.grad.shape, w.data.shape)

    def test_linear_layer(self):
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], name="x")
        layer = Linear(2, 1)
        out = layer(x)
        out.sum().backward()
        self.assertEqual(layer.weight.grad.shape, layer.weight.data.shape)
        self.assertEqual(layer.bias.grad.shape, layer.bias.data.shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)
