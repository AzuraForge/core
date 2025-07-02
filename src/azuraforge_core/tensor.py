import os
from typing import Callable, List, Optional, Set, Tuple, Union, Any
import numpy as np
import logging # Loglama için import et

# === DÜZELTME: Cihaz algılandığında logla (GLOBAL LOGGING BASICCONFIG KALDIRILDI) ===
# Global basicConfig yerine modüle özel logger kullanıyoruz.
logger = logging.getLogger(__name__) # <--- Yeni satır

DEVICE = os.environ.get("AZURAFORGE_DEVICE", "cpu").lower()

xp: Any
if DEVICE == "gpu":
    try:
        import cupy
        xp = cupy
        logger.info("✅ AzuraForge Core: CuPy (GPU) backend successfully loaded.") # <--- logging.info -> logger.info
    except ImportError:
        import numpy
        xp = numpy
        logger.warning("⚠️ AzuraForge Core: AZURAFORGE_DEVICE set to 'gpu' but CuPy not found. Falling back to NumPy (CPU).") # <--- logging.warning -> logger.warning
        DEVICE = "cpu"
else:
    import numpy
    xp = numpy
    logger.info("ℹ️ AzuraForge Core: NumPy (CPU) backend is active.") # <--- logging.info -> logger.info
# === DÜZELTME SONU ===

ArrayType = Any
ScalarType = Union[int, float, bool, np.number, xp.number]

def _empty_backward_op() -> None: pass

class Tensor:
    def __init__(self, data: Any, _children: Tuple["Tensor", ...] = (), _op: str = "", requires_grad: bool = False):
        if isinstance(data, Tensor): self.data = data.data.copy()
        else: 
            # Veriyi doğru cihaza taşı
            try:
                # Eğer xp, cupy ise bu veriyi GPU'ya taşır.
                self.data = xp.array(data, dtype=np.float64)
            except Exception as e:
                # GPU'ya taşıma sırasında hata olursa (örn. CUDA context hatası) logla
                logger.error(f"Error transferring data to device '{DEVICE}': {e}. Falling back to CPU.")
                self.data = np.array(data, dtype=np.float64)

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
            if t.grad is not None:
                t.grad.fill(0.0)
        
        self.grad = xp.ones_like(self.data) if grad_output is None else xp.asarray(grad_output, dtype=np.float64).reshape(self.data.shape)
        
        for v in reversed(topo):
            v._backward()

    def to_cpu(self) -> np.ndarray:
        if hasattr(self.data, 'get'): return self.data.get()
        return np.array(self.data, copy=True)

    def __add__(self, other: Any) -> "Tensor":
        other = _ensure_tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+", self.requires_grad or other.requires_grad)
        def _backward():
            if self.requires_grad and self.grad is not None: self.grad += _unbroadcast_to(self.data.shape, out.grad)
            if other.requires_grad and other.grad is not None: other.grad += _unbroadcast_to(other.data.shape, out.grad)
        out._backward = _backward
        return out

    def __mul__(self, other: Any) -> "Tensor":
        other = _ensure_tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*", self.requires_grad or other.requires_grad)
        def _backward():
            if self.requires_grad and self.grad is not None: self.grad += _unbroadcast_to(self.data.shape, other.data * out.grad)
            if other.requires_grad and other.grad is not None: other.grad += _unbroadcast_to(other.data.shape, self.data * out.grad)
        out._backward = _backward
        return out

    def __pow__(self, power: float) -> "Tensor":
        out = Tensor(self.data ** power, (self,), f"**{power}", self.requires_grad)
        def _backward():
            if self.requires_grad and self.grad is not None: self.grad += (power * (self.data ** (power - 1))) * out.grad
        out._backward = _backward
        return out

    def dot(self, other: "Tensor") -> "Tensor":
        other = _ensure_tensor(other)
        out = Tensor(self.data @ other.data, (self, other), "@", self.requires_grad or other.requires_grad)
        def _backward():
            if self.requires_grad and self.grad is not None: self.grad += out.grad @ other.data.T
            if other.requires_grad and other.grad is not None: other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False) -> "Tensor":
        out = Tensor(xp.sum(self.data, axis=axis, keepdims=keepdims), (self,), "sum", self.requires_grad)
        def _backward(_axis=axis, _keepdims=keepdims):
            if self.requires_grad and self.grad is not None:
                grad_val = out.grad
                if _axis is not None and not _keepdims:
                    grad_val = xp.expand_dims(grad_val, axis=_axis)
                self.grad += xp.ones_like(self.data) * grad_val
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False) -> "Tensor":
        sum_val = self.sum(axis=axis, keepdims=keepdims)
        num_elements = float(np.prod(self.data.shape) / np.prod(sum_val.data.shape))
        return sum_val * (1.0 / num_elements)
    
    def relu(self) -> "Tensor":
        out = Tensor(xp.maximum(0, self.data), (self,), "ReLU", self.requires_grad)
        def _backward():
            if self.requires_grad and self.grad is not None: self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        s = 1 / (1 + xp.exp(-self.data))
        out = Tensor(s, (self,), "Sigmoid", self.requires_grad)
        def _backward():
            if self.requires_grad and self.grad is not None: self.grad += out.data * (1 - out.data) * out.grad
        out._backward = _backward
        return out
    
    def tanh(self) -> "Tensor":
        t = xp.tanh(self.data)
        out = Tensor(t, (self,), "Tanh", self.requires_grad)
        def _backward():
            if self.requires_grad and self.grad is not None:
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

    def sqrt(self) -> "Tensor":
        out = Tensor(xp.sqrt(self.data), (self,), "sqrt", self.requires_grad)
        def _backward():
            if self.requires_grad and self.grad is not None:
                # Derivative of sqrt(x) is 1 / (2 * sqrt(x))
                self.grad += (0.5 / xp.sqrt(self.data)) * out.grad
        out._backward = _backward
        return out
        
def _ensure_tensor(val: Any) -> "Tensor":
    return val if isinstance(val, Tensor) else Tensor(val)

def _unbroadcast_to(target_shape: Tuple[int, ...], grad: ArrayType) -> ArrayType:
    if target_shape == grad.shape:
        return grad
    
    ndim_diff = grad.ndim - len(target_shape)
    if ndim_diff > 0:
        grad = grad.sum(axis=tuple(range(ndim_diff)))

    axes_to_sum = []
    for i, dim in enumerate(target_shape):
        if dim == 1 and grad.shape[i] > 1:
            axes_to_sum.append(i)
    
    if axes_to_sum:
        grad = grad.sum(axis=tuple(axes_to_sum), keepdims=True)
        
    return grad