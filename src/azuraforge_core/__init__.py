# ========== DOSYA: src/azuraforge_core/__init__.py ==========
from .tensor import Tensor, xp, DEVICE, ArrayType, ScalarType, _ensure_tensor

__all__ = ["Tensor", "xp", "DEVICE", "ArrayType", "ScalarType", "_ensure_tensor"]