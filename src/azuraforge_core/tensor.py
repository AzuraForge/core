import os
from typing import Callable, List, Optional, Set, Tuple, Union, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

DEVICE = os.environ.get("AZURAFORGE_DEVICE", "cpu").lower()

xp: Any
if DEVICE == "gpu":
    try:
        import cupy
        xp = cupy
        logger.info("✅ AzuraForge Core: CuPy (GPU) backend successfully loaded.")
    except ImportError:
        import numpy
        xp = numpy
        logger.warning("⚠️ AzuraForge Core: AZURAFORGE_DEVICE set to 'gpu' but CuPy not found. Falling back to NumPy (CPU).")
        DEVICE = "cpu"
else:
    import numpy
    xp = numpy
    logger.info("ℹ️ AzuraForge Core: NumPy (CPU) backend is active.")

ArrayType = Any
ScalarType = Union[int, float, bool, np.number, xp.number]

def _empty_backward_op() -> None: pass

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    N, C, H, W = x_shape
    out_height = (H + 2 * padding - field_height) // stride + 1
    out_width = (W + 2 * padding - field_width) // stride + 1
    i0 = xp.repeat(xp.arange(field_height), field_width)
    i0 = xp.tile(i0, C)
    i1 = stride * xp.repeat(xp.arange(out_height), out_width)
    j0 = xp.tile(xp.arange(field_width), field_height * C)
    j1 = stride * xp.tile(xp.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = xp.repeat(xp.arange(C), field_height * field_width).reshape(-1, 1)
    return (k, i, j)

def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    p = padding
    x_padded = xp.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)
    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols

def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):
    """im2col işleminin tersini uygular."""
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = xp.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    xp.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0: return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]
    
class Tensor:
    # ... (__init__ ve diğer metodlar aynı kalıyor) ...
    def __init__(self, data: Any, _children: Tuple["Tensor", ...] = (), _op: str = "", requires_grad: bool = False):
        if isinstance(data, Tensor): self.data = data.data.copy()
        else: 
            try: self.data = xp.array(data, dtype=xp.float32)
            except Exception as e:
                logger.error(f"Error transferring data to device '{DEVICE}': {e}. Falling back to CPU.")
                self.data = np.array(data, dtype=np.float32)
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
        self.grad = xp.ones_like(self.data) if grad_output is None else xp.asarray(grad_output, dtype=xp.float32).reshape(self.data.shape)
        for v in reversed(topo):
            v._backward()

    def to_cpu(self) -> np.ndarray:
        if hasattr(self.data, 'get'): return self.data.get()
        return np.array(self.data, copy=True)

    def __getitem__(self, indices) -> "Tensor":
        out_data = self.data[indices]
        out = Tensor(out_data, _children=(self,), _op=f"[{indices}]", requires_grad=self.requires_grad)
        def _backward():
            if self.requires_grad and self.grad is not None:
                xp.add.at(self.grad, indices, out.grad)
        out._backward = _backward
        return out

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
                if _axis is not None and not _keepdims: grad_val = xp.expand_dims(grad_val, axis=_axis)
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
            if self.requires_grad and self.grad is not None: self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out
        
    def sqrt(self) -> "Tensor":
        out = Tensor(xp.sqrt(self.data), (self,), "sqrt", self.requires_grad)
        def _backward():
            if self.requires_grad and self.grad is not None: self.grad += (0.5 / xp.sqrt(self.data)) * out.grad
        out._backward = _backward
        return out

    def conv2d(self, weights: "Tensor", bias: "Tensor", stride: int = 1, padding: int = 1) -> "Tensor":
        N, C, H, W = self.data.shape
        F, _, HH, WW = weights.data.shape
        H_out = (H + 2 * padding - HH) // stride + 1
        W_out = (W + 2 * padding - WW) // stride + 1
        
        x_col = im2col_indices(self.data, HH, WW, padding, stride)
        w_row = weights.data.reshape(F, -1)
        
        out_data = w_row @ x_col + bias.data.reshape(-1, 1)
        out_data = out_data.reshape(F, H_out, W_out, N).transpose(3, 0, 1, 2)
        
        out = Tensor(out_data, _children=(self, weights, bias), _op="conv2d", requires_grad=self.requires_grad)

        def _backward():
            if not out.requires_grad or out.grad is None: return

            if bias.requires_grad and bias.grad is not None:
                bias.grad += xp.sum(out.grad, axis=(0, 2, 3))

            if weights.requires_grad and weights.grad is not None:
                db_reshaped = out.grad.transpose(1, 2, 3, 0).reshape(F, -1)
                dw = db_reshaped @ x_col.T
                weights.grad += dw.reshape(weights.data.shape)

            if self.requires_grad and self.grad is not None:
                dout_reshaped = out.grad.transpose(1, 2, 3, 0).reshape(F, -1)
                dx_col = w_row.T @ dout_reshaped
                self.grad += col2im_indices(dx_col, self.data.shape, HH, WW, padding, stride)

        out._backward = _backward
        return out
        
    def max_pool2d(self, kernel_size: int = 2, stride: int = 2) -> "Tensor":
        N, C, H, W = self.data.shape
        HH, WW = kernel_size, kernel_size
        H_out = (H - HH) // stride + 1
        W_out = (W - WW) // stride + 1

        x_reshaped = self.data.reshape(N * C, 1, H, W)
        x_col = im2col_indices(x_reshaped, HH, WW, padding=0, stride=stride)
        
        out_data = xp.max(x_col, axis=0)
        out_data = out_data.reshape(H_out, W_out, N, C).transpose(2, 3, 0, 1)
        
        out = Tensor(out_data, _children=(self,), _op="max_pool2d", requires_grad=self.requires_grad)

        def _backward():
            if not out.requires_grad or out.grad is None: return
            
            x_col_max_val_indices = xp.argmax(x_col, axis=0)
            dx_col = xp.zeros_like(x_col)
            dout_flat = out.grad.transpose(2, 3, 0, 1).ravel()
            dx_col[x_col_max_val_indices, range(dout_flat.size)] = dout_flat
            
            dx = col2im_indices(dx_col, x_reshaped.shape, HH, WW, padding=0, stride=stride)
            self.grad += dx.reshape(self.data.shape)

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
    if target_shape == grad.shape: return grad
    ndim_diff = grad.ndim - len(target_shape)
    if ndim_diff > 0: grad = grad.sum(axis=tuple(range(ndim_diff)))
    axes_to_sum = [i for i, dim in enumerate(target_shape) if dim == 1 and grad.shape[i] > 1]
    if axes_to_sum: grad = grad.sum(axis=tuple(axes_to_sum), keepdims=True)
    return grad