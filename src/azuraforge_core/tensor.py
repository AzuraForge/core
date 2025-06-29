import os
from typing import Callable, List, Optional, Set, Tuple, Union, Any, cast
import numpy as np

DEVICE = os.environ.get("AZURAFORGE_DEVICE", "cpu").lower()

xp: Any
if DEVICE == "gpu":
    try:
        import cupy
        xp = cupy
    except ImportError:
        import numpy
        xp = numpy
        DEVICE = "cpu"
else:
    import numpy
    xp = numpy

ArrayType = Any
ScalarType = Union[int, float, bool, np.number, xp.number]

def _empty_backward_op() -> None: pass

class Tensor:
    def __init__(self, data: Any, _children: Tuple["Tensor", ...] = (), _op: str = "", requires_grad: bool = False):
        if isinstance(data, Tensor): self.data = data.data.copy()
        else: self.data = xp.array(data, dtype=np.float64)
        
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
        self.grad = xp.ones_like(self.data) if grad_output is None else xp.asarray(grad_output, dtype=np.float64).reshape(self.data.shape)
        for v in reversed(topo): v._backward()

    def to_cpu(self) -> np.ndarray:
        if hasattr(self.data, 'get'): return self.data.get()
        return np.array(self.data, copy=True)

    def __add__(self, other: Any) -> "Tensor":
        other = _ensure_tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+", self.requires_grad or other.requires_grad)
        def _backward():
            if self.requires_grad: self.grad += _unbroadcast_to(self.data.shape, out.grad)
            if other.requires_grad: other.grad += _unbroadcast_to(other.data.shape, out.grad)
        out._backward = _backward
        return out

    def __mul__(self, other: Any) -> "Tensor":
        other = _ensure_tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*", self.requires_grad or other.requires_grad)
        def _backward():
            if self.requires_grad: self.grad += _unbroadcast_to(self.data.shape, other.data * out.grad)
            if other.requires_grad: other.grad += _unbroadcast_to(other.data.shape, self.data * out.grad)
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
        def _backward(_axis=axis, _keepdims=keepdims):
            if self.requires_grad and self.grad is not None:
                grad_val = out.grad
                if _axis is not None and not _keepdims:
                    grad_val = xp.expand_dims(grad_val, axis=_axis)
                self.grad += grad_val
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False) -> "Tensor":
        sum_val = self.sum(axis=axis, keepdims=keepdims)
        num_elements = float(np.prod(self.data.shape) / np.prod(sum_val.data.shape))
        return sum_val * (1.0 / num_elements)
    
    def relu(self) -> "Tensor":
        out = Tensor(xp.maximum(0, self.data), (self,), "ReLU", self.requires_grad)
        def _backward():
            if self.requires_grad: self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        s = 1 / (1 + xp.exp(-self.data))
        out = Tensor(s, (self,), "Sigmoid", self.requires_grad)
        def _backward():
            if self.requires_grad: self.grad += out.data * (1 - out.data) * out.grad
        out._backward = _backward
        return out
    
    # YENİ: Tanh aktivasyon fonksiyonu
    def tanh(self) -> "Tensor":
        t = xp.tanh(self.data)
        out = Tensor(t, (self,), "Tanh", self.requires_grad)
        def _backward():
            if self.requires_grad:
                # Gradyan: (1 - tanh(x)^2) * grad_output
                # t = tanh(self.data) olduğu için, (1 - t^2) kullanabiliriz.
                self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out
        
    def __repr__(self): return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __truediv__(self, other): return self * (_ensure_tensor(other) ** -1)
    __radd__ = __add__
    def __rmul__(self, other): return self * other
    def __rsub__(self, other): return _ensure_tensor(other) - self
    def __rtruediv__(self, other): return _ensure_tensor(other) / self

def _ensure_tensor(val: Any) -> "Tensor":
    return val if isinstance(val, Tensor) else Tensor(val)

def _unbroadcast_to(target_shape: Tuple[int, ...], grad: ArrayType) -> ArrayType:
    """Bir gradyanı, orijinal tensörün (yayınlamadan önceki) şekline geri küçültür."""
    if target_shape == grad.shape:
        return grad
    
    # Boyut sayısını eşitle
    ndim_diff = grad.ndim - len(target_shape)
    if ndim_diff > 0:
        grad = grad.sum(axis=tuple(range(ndim_diff)))

    # Boyutu 1 olan eksenler boyunca topla
    axes_to_sum = []
    for i, dim in enumerate(target_shape):
        if dim == 1:
            axes_to_sum.append(i)
    
    if axes_to_sum:
        grad = grad.sum(axis=tuple(axes_to_sum), keepdims=True)
        
    return grad