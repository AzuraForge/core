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

def _empty_backward_op() -> None:
    pass

class Tensor:
    """
    Otomatik türev yeteneğine sahip çok boyutlu bir dizi (tensor).
    AzuraForge ekosisteminin temel yapı taşıdır.
    """
    data: ArrayType
    grad: Optional[ArrayType]
    requires_grad: bool
    _backward: Callable[[], None]
    _prev: Set["Tensor"]
    _op: str

    def __init__(
        self,
        data: Union[ScalarType, list, np.ndarray, xp.ndarray, "Tensor"],
        _children: Tuple["Tensor", ...] = (),
        _op: str = "",
        requires_grad: bool = False,
    ):
        if isinstance(data, Tensor):
            self.data = xp.asarray(data.data, dtype=xp.float64).copy()
        else:
            self.data = xp.array(data, dtype=xp.float64)
        
        self.requires_grad = requires_grad
        self.grad = xp.zeros_like(self.data) if self.requires_grad else None
        
        self._backward = _empty_backward_op
        self._prev = set(_children)
        self._op = _op

    def to_cpu(self) -> np.ndarray:
        """Tensor verisini her zaman bir NumPy dizisi olarak CPU'ya döndürür."""
        if DEVICE == "gpu" and hasattr(xp, "asnumpy"):
            return cast(np.ndarray, xp.asnumpy(self.data).astype(np.float64))
        return cast(np.ndarray, np.array(self.data, dtype=np.float64, copy=True))

    def backward(self, grad_output: Optional[ArrayType] = None) -> None:
        """Bu tensörden başlayarak geriye doğru yayılım yaparak gradyanları hesaplar."""
        if not self.requires_grad:
            return

        topo: List[Tensor] = []
        visited: Set[Tensor] = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)

        for t in topo:
            if t.grad is not None:
                t.grad.fill(0.0)

        self.grad = xp.ones_like(self.data) if grad_output is None else xp.asarray(grad_output, dtype=xp.float64)
        
        for v in reversed(topo):
            v._backward()

    # --- Operatörler ve Matematiksel Fonksiyonlar ---
    def __add__(self, other: Union["Tensor", ScalarType]) -> "Tensor":
        other = _ensure_tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+", self.requires_grad or other.requires_grad)

        def _backward():
            if out.grad is not None:
                if self.grad is not None: self.grad += _unbroadcast(self.data, out.grad)
                if other.grad is not None: other.grad += _unbroadcast(other.data, out.grad)
        out._backward = _backward
        return out

    def __mul__(self, other: Union["Tensor", ScalarType]) -> "Tensor":
        other = _ensure_tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*", self.requires_grad or other.requires_grad)

        def _backward():
            if out.grad is not None:
                if self.grad is not None: self.grad += other.data * out.grad
                if other.grad is not None: other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, power: ScalarType) -> "Tensor":
        out = Tensor(self.data ** float(power), (self,), f"**{power}", self.requires_grad)

        def _backward():
            if out.grad is not None and self.grad is not None:
                self.grad += (float(power) * (self.data ** (float(power) - 1))) * out.grad
        out._backward = _backward
        return out

    def dot(self, other: "Tensor") -> "Tensor":
        other = _ensure_tensor(other)
        out = Tensor(self.data @ other.data, (self, other), "dot", self.requires_grad or other.requires_grad)

        def _backward():
            if out.grad is not None:
                if self.grad is not None: self.grad += out.grad @ other.data.T
                if other.grad is not None: other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    def relu(self) -> "Tensor":
        out = Tensor(xp.maximum(0, self.data), (self,), "ReLU", self.requires_grad)

        def _backward():
            if out.grad is not None and self.grad is not None:
                self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False) -> "Tensor":
        out = Tensor(xp.sum(self.data, axis=axis, keepdims=keepdims), (self,), "sum", self.requires_grad)
        
        def _backward():
            if out.grad is not None and self.grad is not None:
                output_grad = out.grad
                if axis is not None and not keepdims:
                    output_grad = xp.expand_dims(out.grad, axis)
                self.grad += xp.ones_like(self.data) * output_grad
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False) -> "Tensor":
        sum_val = self.sum(axis=axis, keepdims=keepdims)
        num_elements = float(np.prod(self.data.shape) / np.prod(sum_val.data.shape))
        return sum_val * (1.0 / num_elements)

    # --- Yardımcı Metodlar ve Diğer Operatörler ---
    def __repr__(self): return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return _ensure_tensor(other) - self
    def __truediv__(self, other): return self * (_ensure_tensor(other) ** -1)
    def __rtruediv__(self, other): return _ensure_tensor(other) * (self ** -1)

def _ensure_tensor(val: Any) -> "Tensor":
    return val if isinstance(val, Tensor) else Tensor(val)

def _unbroadcast(tensor_data: ArrayType, grad_data: ArrayType) -> ArrayType:
    if tensor_data.shape == grad_data.shape:
        return grad_data
    
    # Boyut sayısını eşitlemek için gradyanın başına eksen ekle
    ndim_to_sum = grad_data.ndim - tensor_data.ndim
    if ndim_to_sum > 0:
        grad_data = grad_data.sum(axis=tuple(range(ndim_to_sum)))

    # Boyutu 1 olan eksenler boyunca topla
    axes_to_sum = [i for i, dim in enumerate(tensor_data.shape) if dim == 1]
    if axes_to_sum:
        grad_data = grad_data.sum(axis=tuple(axes_to_sum), keepdims=True)
        
    return grad_data