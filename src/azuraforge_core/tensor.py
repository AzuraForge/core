# ========== GÜNCELLENECEK DOSYA: core/src/azuraforge_core/tensor.py ==========
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
        print(f"AzuraForge/core: CuPy (GPU) backend enabled. (Device: {cupy.cuda.runtime.getDevice()})")
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

def _ensure_tensor(val: Any) -> "Tensor":
    """Verilen değeri bir Tensor nesnesi değilse, bir Tensor'a dönüştürür."""
    return val if isinstance(val, Tensor) else Tensor(val)

class Tensor:
    """
    Otomatik türev yeteneğine sahip çok boyutlu bir dizi (tensor).
    AzuraForge ekosisteminin temel yapı taşıdır.
    """
    def __init__(self, data: Any, _children: Tuple["Tensor", ...] = (), _op: str = "", requires_grad: bool = False):
        if isinstance(data, Tensor):
            self.data = data.data.copy()
        else:
            self.data = xp.array(data, dtype=np.float64)
        
        self.requires_grad = requires_grad
        self.grad: Optional[ArrayType] = xp.zeros_like(self.data) if requires_grad else None
        self._backward: Callable[[], None] = _empty_backward_op
        self._prev: Set["Tensor"] = set(_children)
        self._op: str = _op

    def backward(self, grad_output: Optional[ArrayType] = None) -> None:
        """Bu tensörden başlayarak geriye doğru yayılım yaparak gradyanları hesaplar."""
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
        """Tensor verisini her zaman bir NumPy dizisi olarak CPU'ya döndürür."""
        if hasattr(self.data, 'get'): # CuPy dizileri .get() metoduna sahiptir
            return self.data.get()
        return np.array(self.data, copy=True)

    # --- Operatörler ve Matematiksel Fonksiyonlar ---
    def __add__(self, other: Any) -> "Tensor":
        other = _ensure_tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+", self.requires_grad or other.requires_grad)
        def _backward():
            if self.requires_grad: self.grad += out.grad
            if other.requires_grad: other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other: Any) -> "Tensor":
        other = _ensure_tensor(other)
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
        
    def __repr__(self): return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __truediv__(self, other): return self * (_ensure_tensor(other) ** -1)
    __radd__ = __add__
    def __rmul__(self, other): return self * other
    def __rsub__(self, other): return _ensure_tensor(other) - self
    def __rtruediv__(self, other): return _ensure_tensor(other) / self