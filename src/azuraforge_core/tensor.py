# ========== GÜNCELLENECEK DOSYA: core/src/azuraforge_core/tensor.py ==========
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
        else: self.data = xp.array(data, dtype=xp.float64)
        
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

    # --- YENİ EKLENEN METOT ---
    def to_cpu(self) -> np.ndarray:
        """Tensor verisini her zaman bir NumPy dizisi olarak CPU'ya döndürür."""
        if DEVICE == "gpu" and hasattr(xp, "asnumpy"):
            return cast(np.ndarray, xp.asnumpy(self.data))
        return cast(np.ndarray, np.array(self.data, copy=True))

    # --- Diğer Metotlar ---
    def __add__(self, other: Any) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+", self.requires_grad or other.requires_grad)
        def _backward():
            if self.requires_grad: self.grad += _unbroadcast(self.data, out.grad)
            if other.requires_grad: other.grad += _unbroadcast(other.data, out.grad)
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
                    # Bu eksenleri orijinal şekle geri ekle
                    shape = list(self.data.shape)
                    if isinstance(axis, int): axis = (axis,)
                    for i in sorted(axis): shape[i] = 1
                    grad = grad.reshape(shape)
                self.grad += grad
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
        
    def __repr__(self): return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __truediv__(self, other): return self * ((other if isinstance(other, Tensor) else Tensor(other)) ** -1)
    __radd__ = __add__
    def __rmul__(self, other): return self * other
    def __rsub__(self, other): return Tensor(other) - self
    def __rtruediv__(self, other): return Tensor(other) / self

def _unbroadcast(target_data: ArrayType, grad: ArrayType) -> ArrayType:
    if target_data.shape == grad.shape: return grad
    # ... (Unbroadcast mantığı) ...
    return grad