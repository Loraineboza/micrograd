import numpy as np


def _as_shape_tuple(shape):
    if isinstance(shape, int):
        return (shape,)
    return tuple(shape)


def _reduce_broadcast_grad(grad, target_shape):
    """Reduce a gradient back to the shape of a broadcasted tensor.

    This helper is used when one operand is a vector or scalar and is
    broadcast to a larger array during an elementwise operation. The output
    gradient is summed along axes that were introduced by broadcasting.
    """
    grad = np.asarray(grad, dtype=np.float64)
    target_shape = tuple(target_shape)

    if grad.shape == target_shape:
        return grad

    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)

    if grad.ndim < len(target_shape):
        pad = len(target_shape) - grad.ndim
        grad = grad.reshape((1,) * pad + grad.shape)

    for axis in range(len(target_shape)):
        if target_shape[axis] == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad.reshape(target_shape)


class Tensor:
    def __init__(self, data, requires_grad=True, name=None, parents=()):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        self.parents = tuple(parents)
        self._backward = lambda: None
        self.requires_grad = requires_grad
        self.name = name or "tensor"
        self.op = ""

    @property
    def shape(self):
        return self.data.shape

    @staticmethod
    def _ensure_tensor(value):
        if isinstance(value, Tensor):
            return value
        return Tensor(np.asarray(value, dtype=np.float64), requires_grad=False)

    def zero_grad(self):
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        for parent in self.parents:
            if isinstance(parent, Tensor):
                parent.zero_grad()

    def __repr__(self):
        return f"Tensor(name={self.name!r}, data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = self._ensure_tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            name=f"{self.name}+{other.name}",
            parents=(self, other),
        )
        out.op = "+"

        def backward():
            if self.requires_grad:
                self.grad += _reduce_broadcast_grad(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _reduce_broadcast_grad(out.grad, other.data.shape)

        out._backward = backward
        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        other = self._ensure_tensor(other)
        out = Tensor(
            self.data - other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            name=f"{self.name}-{other.name}",
            parents=(self, other),
        )
        out.op = "-"

        def backward():
            if self.requires_grad:
                self.grad += _reduce_broadcast_grad(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad -= _reduce_broadcast_grad(out.grad, other.data.shape)

        out._backward = backward
        return out

    def __rsub__(self, other):
        return self._ensure_tensor(other) - self

    def __mul__(self, other):
        other = self._ensure_tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            name=f"{self.name}*{other.name}",
            parents=(self, other),
        )
        out.op = "*"

        def backward():
            if self.requires_grad:
                self.grad += _reduce_broadcast_grad(other.data * out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _reduce_broadcast_grad(self.data * out.grad, other.data.shape)

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
            out = Tensor(
                self.data ** power.data,
                requires_grad=self.requires_grad or power.requires_grad,
                name=f"{self.name}**{power.name}",
                parents=(self, power),
            )

            def backward():
                if self.requires_grad:
                    self.grad += _reduce_broadcast_grad(
                        power.data * (self.data ** (power.data - 1)) * out.grad,
                        self.data.shape,
                    )
                if power.requires_grad:
                    power.grad += _reduce_broadcast_grad(
                        np.log(np.abs(self.data)) * out.grad * (self.data ** power.data),
                        power.data.shape,
                    )

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
        if self.data.ndim != 2 or other.data.ndim != 2:
            raise ValueError(
                f"Tensor.__matmul__ supports only 2D arrays; got shapes {self.data.shape} and {other.data.shape}."
            )
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
                grad = np.ones_like(self.data, dtype=np.float64) * out.grad
            else:
                grad = np.asarray(out.grad, dtype=np.float64)
                axes = sorted(_as_shape_tuple(axis), reverse=True)
                if not keepdims:
                    shape = list(self.data.shape)
                    for ax in _as_shape_tuple(axis):
                        shape[ax] = 1
                    grad = grad.reshape(shape)
                for ax in axes:
                    if keepdims:
                        grad = np.expand_dims(grad, axis=ax)
                if not keepdims:
                    grad = np.broadcast_to(grad, self.data.shape)
                else:
                    grad = np.broadcast_to(grad, self.data.shape)
            self.grad += grad

        out._backward = backward
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            size = self.data.size
        else:
            axes = _as_shape_tuple(axis)
            size = np.prod([self.data.shape[a] for a in axes], dtype=np.float64)
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
                if isinstance(parent, Tensor):
                    dfs(parent)
            topo.append(node)

        dfs(self)

        self.grad = self.grad + np.ones_like(self.data, dtype=np.float64)

        for node in reversed(topo):
            if node.requires_grad:
                node._backward()


__all__ = ["Tensor"]


# NOTE: we intentionally keep the original API shape and behaviour, while making
# broadcasting-aware gradient accumulation explicit and reproducible.
