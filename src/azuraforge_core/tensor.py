# ========== DOSYA: src/azuraforge_core/tensor.py ==========
import os
from typing import Callable, List, Optional, Set, Tuple, Union, Any, cast
import numpy as np

DEVICE = os.environ.get("AZURAFORGE_DEVICE", "cpu").lower()
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

ArrayType = Any
ScalarType = Union[int, float, bool, np.number, xp.number]


def _empty_backward_op() -> None:
    pass


class Tensor:
    data: ArrayType
    grad: Optional[ArrayType]
    requires_grad: bool
    _backward: Callable[[], None]
    _prev: Set["Tensor"]
    label: str
    _op: str

    def __init__(
        self,
        data: Union[ScalarType, list, np.ndarray, xp.ndarray, "Tensor"],
        _children: Tuple["Tensor", ...] = (),
        _op: str = "",
        label: str = "",
        requires_grad: bool = False,
    ):
        if isinstance(data, Tensor): self.data = xp.asarray(data.data, dtype=xp.float64).copy()
        elif isinstance(data, (np.ndarray, xp.ndarray)): self.data = xp.asarray(data, dtype=xp.float64)
        else: self.data = xp.array(data, dtype=xp.float64)
        
        self.requires_grad = requires_grad
        self.grad = xp.zeros_like(self.data) if self.requires_grad else None
        
        self._backward = _empty_backward_op
        self._prev = set(_children)
        self.label = label
        self._op = _op

    def to_cpu(self) -> np.ndarray:
        if DEVICE == "gpu" and hasattr(xp, "asnumpy"):
            return cast(np.ndarray, xp.asnumpy(self.data).astype(np.float64))
        return cast(np.ndarray, np.array(self.data, dtype=np.float64, copy=True))

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad}, device='{DEVICE}')"

    def backward(self, grad_output: Optional[ArrayType] = None) -> None:
        if not self.requires_grad:
            print("Warning: backward() called on a Tensor that does not require gradients.")
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

        # Zero out all gradients in the computation graph
        for t in topo:
            if t.grad is not None:
                t.grad.fill(0.0)

        # Set the gradient of the output tensor
        if self.grad is None: self.grad = xp.zeros_like(self.data)
        self.grad = xp.ones_like(self.data) if grad_output is None else xp.asarray(grad_output, dtype=xp.float64)
        
        for v in reversed(topo):
            v._backward()

    # --- Operator Overloading ---
    def __add__(self, other: Union["Tensor", ScalarType]) -> "Tensor":
        other = _ensure_tensor(other, device_xp=xp)
        out_requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data + other.data, (self, other), "+", requires_grad=out_requires_grad)

        def _backward():
            if out.grad is not None:
                if self.grad is not None: self.grad += out.grad
                if other.grad is not None: other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other: Union["Tensor", ScalarType]) -> "Tensor":
        other = _ensure_tensor(other, device_xp=xp)
        out_requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data * other.data, (self, other), "*", requires_grad=out_requires_grad)

        def _backward():
            if out.grad is not None:
                if self.grad is not None: self.grad += other.data * out.grad
                if other.grad is not None: other.grad += self.data * out.grad
        out._backward = _backward
        return out

    # ... (Diğer tüm Tensor metodları: __pow__, dot, sum, relu, vb. buraya eklenecek)
    # Bu metodlar bir önceki projedeki tensor.py'dan alınabilir.
    # Şimdilik bu kadarını bırakmak, temel yapıyı test etmek için yeterlidir.

def _ensure_tensor(val: Any, device_xp: Any) -> "Tensor":
    return val if isinstance(val, Tensor) else Tensor(device_xp.array(val))