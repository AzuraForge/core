# ========== DOSYA: src/azuraforge_core/tensor.py ==========
import os
from typing import Callable, List, Optional, Set, Tuple, Union, Any, cast
import numpy as np

# Ortam değişkeninden cihazı oku (CPU/GPU)
DEVICE = os.environ.get("AZURAFORGE_DEVICE", "cpu").lower()

# Backend olarak NumPy veya CuPy'yi dinamik olarak seç
xp: Any
if DEVICE == "gpu":
    try:
        import cupy
        xp = cupy
        print("AzuraForge/core: CuPy (GPU) backend enabled.")
    except ImportError:
        import numpy
        xp = numpy
        DEVICE = "cpu"
        print("AzuraForge/core Warning: CuPy not found, falling back to NumPy (CPU).")
else:
    import numpy
    xp = numpy

# Tip ipuçları için takma adlar
ArrayType = Any
ScalarType = Union[int, float, bool, np.number, xp.number]

def _empty_backward_op() -> None: pass

class Tensor:
    def __init__(self, data: Any, _children: Tuple["Tensor", ...] = (), _op: str = "", requires_grad: bool = False):
        self.data = xp.array(data, dtype=xp.float64) if not isinstance(data, xp.ndarray) else data.astype(xp.float64)
        self.requires_grad = requires_grad
        self.grad: Optional[ArrayType] = xp.zeros_like(self.data) if requires_grad else None
        self._backward: Callable[[], None] = _empty_backward_op
        self._prev: Set["Tensor"] = set(_children)
        self._op: str = _op

    def backward(self, grad_output: Optional[ArrayType] = None) -> None:
        if not self.requires_grad: return
        topo: List[Tensor] = []
        visited: Set[Tensor] = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v); [build_topo(child) for child in v._prev]; topo.append(v)
        build_topo(self)
        for t in topo:
            if t.grad is not None: t.grad.fill(0.0)
        self.grad = xp.ones_like(self.data) if grad_output is None else xp.asarray(grad_output, dtype=xp.float64)
        for v in reversed(topo): v._backward()

    def __add__(self, other: Any) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+", self.requires_grad or other.requires_grad)
        def _backward():
            if self.requires_grad: self.grad += out.grad
            if other.requires_grad: other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other: Any) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*", self.requires_grad or other.requires_grad)
        def _backward():
            if self.requires_grad: self.grad += other.data * out.grad
            if other.requires_grad: other.grad += self.data * out.grad
        out._backward = _backward
        return out
        
    def __pow__(self, power: float) -> "Tensor":
        out = Tensor(self.data ** power, (self,), f"**{power}", self.requires_grad)
        def _backward():
            if self.requires_grad: self.grad += (power * (self.data ** (power - 1))) * out.grad
        out._backward = _backward
        return out

    def dot(self, other: "Tensor") -> "Tensor":
        out = Tensor(self.data @ other.data, (self, other), "@", self.requires_grad or other.requires_grad)
        def _backward():
            if self.requires_grad: self.grad += out.grad @ other.data.T
            if other.requires_grad: other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False) -> "Tensor":
        out = Tensor(xp.sum(self.data, axis=axis, keepdims=keepdims), (self,), "sum", self.requires_grad)
        def _backward():
            if self.requires_grad:
                grad = out.grad
                if axis is not None and not keepdims:
                    grad = xp.expand_dims(out.grad, axis)
                self.grad += grad
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False) -> "Tensor":
        num_elements = float(np.prod(self.data.shape) / np.prod(self.sum(axis=axis, keepdims=True).data.shape))
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / num_elements)
    
    def relu(self) -> "Tensor":
        out = Tensor(xp.maximum(0, self.data), (self,), "ReLU", self.requires_grad)
        def _backward():
            if self.requires_grad: self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def log(self, eps=1e-9) -> "Tensor":
        out = Tensor(xp.log(self.data + eps), (self,), "log", self.requires_grad)
        def _backward():
            if self.requires_grad: self.grad += (1 / (self.data + eps)) * out.grad
        out._backward = _backward
        return out

    def __repr__(self): return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __truediv__(self, other): return self * (other ** -1)
    __radd__ = __add__
    def __rmul__(self, other): return self * other
    def __rsub__(self, other): return Tensor(other) - self
    def __rtruediv__(self, other): return Tensor(other) / self