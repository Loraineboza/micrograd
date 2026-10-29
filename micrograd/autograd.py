import numpy as np


def _as_shape_tuple(shape):
    if isinstance(shape, int):
        return (shape,)
    return tuple(shape)


class Tensor:
    def __init__(self, data, requires_grad=True, name=None, parents=()):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        self.parents = tuple(parents)
        self._backward = lambda: None
        self.requires_grad = requires_grad
        self.name = name or "tensor"
        self.op = ""

    @staticmethod
    def _ensure_tensor(value):
        if isinstance(value, Tensor):
            return value
        return Tensor(np.asarray(value, dtype=np.float64), requires_grad=False)

    def zero_grad(self):
        self.grad = np.zeros_like(self.data, dtype=np.float64)

    def __repr__(self):
        return f"Tensor(name={self.name!r}, data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = self._ensure_tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad, name=f"{self.name}+{other.name}", parents=(self, other))
        out.op = "+"

        def backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad

        out._backward = backward
        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        other = self._ensure_tensor(other)
        out = Tensor(self.data - other.data, requires_grad=self.requires_grad or other.requires_grad, name=f"{self.name}-{other.name}", parents=(self, other))
        out.op = "-"

        def backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad -= out.grad

        out._backward = backward
        return out

    def __rsub__(self, other):
        return self._ensure_tensor(other) - self

    def __mul__(self, other):
        other = self._ensure_tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad, name=f"{self.name}*{other.name}", parents=(self, other))
        out.op = "*"

        def backward():
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                other.grad += self.data * out.grad

        out._backward = backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (self._ensure_tensor(other) ** -1)

    def __rtruediv__(self, other):
        return self._ensure_tensor(other) / self

    def __neg__(self):
        return self * -1

    def __pow__(self, power):
        if not isinstance(power, (int, float, np.number)):
            power = self._ensure_tensor(power)
            out = Tensor(self.data ** power.data, requires_grad=self.requires_grad or power.requires_grad, name=f"{self.name}**{power.name}", parents=(self, power))

            def backward():
                if self.requires_grad:
                    self.grad += power.data * (self.data ** (power.data - 1)) * out.grad
                if power.requires_grad:
                    power.grad += np.log(np.abs(self.data)) * out.grad * (self.data ** power.data)

            out._backward = backward
            return out

        out = Tensor(self.data ** power, requires_grad=self.requires_grad, name=f"{self.name}**{power}", parents=(self,))

        def backward():
            if self.requires_grad:
                self.grad += power * (self.data ** (power - 1)) * out.grad

        out._backward = backward
        return out

    def __matmul__(self, other):
        other = self._ensure_tensor(other)
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad, name=f"{self.name}@{other.name}", parents=(self, other))
        out.op = "@"

        def backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad

        out._backward = backward
        return out

    def __rmatmul__(self, other):
        other = self._ensure_tensor(other)
        return other @ self

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, name=f"sum({self.name})", parents=(self,))
        out.op = "sum"

        def backward():
            if not self.requires_grad:
                return
            if axis is None:
                grad = np.ones_like(self.data) * out.grad
            else:
                grad = np.expand_dims(out.grad, axis=axis) if not keepdims else out.grad
                if not keepdims:
                    shape = list(self.data.shape)
                    for ax in _as_shape_tuple(axis):
                        shape[ax] = 1
                    grad = np.reshape(grad, shape)
                grad = np.broadcast_to(grad, self.data.shape)
            self.grad += grad

        out._backward = backward
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            size = self.data.size
        else:
            if isinstance(axis, (tuple, list)):
                size = np.prod([self.data.shape[a] for a in axis])
            else:
                size = self.data.shape[axis]
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, name=f"mean({self.name})", parents=(self,))
        out.op = "mean"

        def backward():
            if not self.requires_grad:
                return
            grad = out.grad / size
            if axis is None:
                self.grad += np.ones_like(self.data) * grad
            else:
                if keepdims:
                    self.grad += np.broadcast_to(grad, self.data.shape)
                else:
                    shape = list(self.data.shape)
                    for ax in _as_shape_tuple(axis):
                        shape[ax] = 1
                    grad = np.reshape(grad, shape)
                    self.grad += np.broadcast_to(grad, self.data.shape)

        out._backward = backward
        return out

    def reshape(self, shape):
        out = Tensor(self.data.reshape(shape), requires_grad=self.requires_grad, name=f"reshape({self.name})", parents=(self,))
        out.op = "reshape"

        def backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.data.shape)

        out._backward = backward
        return out

    def T(self):
        out = Tensor(self.data.T, requires_grad=self.requires_grad, name=f"transpose({self.name})", parents=(self,))
        out.op = "T"

        def backward():
            if self.requires_grad:
                self.grad += out.grad.T

        out._backward = backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), requires_grad=self.requires_grad, name=f"exp({self.name})", parents=(self,))
        out.op = "exp"

        def backward():
            if self.requires_grad:
                self.grad += out.grad * out.data

        out._backward = backward
        return out

    def log(self):
        safe = np.clip(self.data, 1e-7, None)
        out = Tensor(np.log(safe), requires_grad=self.requires_grad, name=f"log({self.name})", parents=(self,))
        out.op = "log"

        def backward():
            if self.requires_grad:
                self.grad += out.grad / safe

        out._backward = backward
        return out

    def sigmoid(self):
        out = Tensor(1.0 / (1.0 + np.exp(-self.data)), requires_grad=self.requires_grad, name=f"sigmoid({self.name})", parents=(self,))
        out.op = "sigmoid"

        def backward():
            if self.requires_grad:
                self.grad += out.grad * out.data * (1.0 - out.data)

        out._backward = backward
        return out

    def tanh(self):
        out = Tensor(np.tanh(self.data), requires_grad=self.requires_grad, name=f"tanh({self.name})", parents=(self,))
        out.op = "tanh"

        def backward():
            if self.requires_grad:
                self.grad += out.grad * (1.0 - np.tanh(self.data) ** 2)

        out._backward = backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0.0, self.data), requires_grad=self.requires_grad, name=f"relu({self.name})", parents=(self,))
        out.op = "relu"

        def backward():
            if self.requires_grad:
                self.grad += out.grad * (self.data > 0).astype(np.float64)

        out._backward = backward
        return out

    def clip(self, min_value, max_value):
        out = Tensor(np.clip(self.data, min_value, max_value), requires_grad=self.requires_grad, name=f"clip({self.name})", parents=(self,))
        out.op = "clip"

        def backward():
            if self.requires_grad:
                mask = (self.data >= min_value) & (self.data <= max_value)
                self.grad += out.grad * mask.astype(np.float64)

        out._backward = backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def dfs(node):
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in node.parents:
                if parent is not None:
                    dfs(parent)
            topo.append(node)

        dfs(self)

        self.grad = np.ones_like(self.data, dtype=np.float64)
        for node in reversed(topo):
            if node.requires_grad:
                node._backward()


__all__ = ["Tensor"]
