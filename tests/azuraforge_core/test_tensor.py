import pytest
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from azuraforge_core import Tensor, xp

# --- Temel Testler ---
def test_tensor_creation_and_defaults():
    t = Tensor([1, 2, 3])
    assert isinstance(t.data, xp.ndarray)
    assert t.requires_grad is False
    assert t.grad is None
    assert t.data.dtype == xp.float32

# --- Gradyan Doğruluk Testleri (PyTorch ile Karşılaştırmalı) ---
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed, skipping gradient checks.")
@pytest.mark.parametrize("op", [
    "add", "sub", "mul", "div", "pow", "dot", "sum", "mean", 
    "relu", "sigmoid", "tanh", "getitem", "conv2d", "max_pool2d"
])
def test_gradient_correctness_vs_pytorch(op):
    """Farklı operasyonlar için gradyanların PyTorch sonuçlarıyla eşleştiğini doğrular."""
    
    # İşlemlere göre tensörleri hazırla
    if op == "dot":
        a_data = np.random.randn(2, 3).astype(np.float32)
        b_data = np.random.randn(3, 4).astype(np.float32)
        a_az, b_az = Tensor(a_data, requires_grad=True), Tensor(b_data, requires_grad=True)
        a_th, b_th = torch.tensor(a_data, requires_grad=True), torch.tensor(b_data, requires_grad=True)
        out_az, out_th = a_az.dot(b_az).sum(), (a_th @ b_th).sum()
    elif op == "conv2d":
        a_data = np.random.randn(2, 3, 10, 10).astype(np.float32)
        b_data = np.random.randn(4, 3, 3, 3).astype(np.float32)
        c_data = np.random.randn(4).astype(np.float32)
        a_az, b_az, c_az = Tensor(a_data, requires_grad=True), Tensor(b_data, requires_grad=True), Tensor(c_data, requires_grad=True)
        a_th, b_th, c_th = torch.tensor(a_data, requires_grad=True), torch.tensor(b_data, requires_grad=True), torch.tensor(c_data, requires_grad=True)
        out_az, out_th = a_az.conv2d(b_az, c_az).sum(), torch.nn.functional.conv2d(a_th, b_th, c_th, padding=1).sum()
    elif op == "max_pool2d":
        a_data = np.random.randn(2, 3, 10, 10).astype(np.float32)
        a_az, a_th = Tensor(a_data, requires_grad=True), torch.tensor(a_data, requires_grad=True)
        out_az, out_th = a_az.max_pool2d().sum(), torch.nn.functional.max_pool2d(a_th, kernel_size=2, stride=2).sum()
    elif op == "getitem":
        a_data = np.random.randn(10, 5).astype(np.float32)
        indices = np.random.randint(0, 10, size=8)
        a_az, a_th = Tensor(a_data, requires_grad=True), torch.tensor(a_data, requires_grad=True)
        out_az, out_th = a_az[indices].sum(), a_th[indices].sum()
    else: # Diğer tüm ikili işlemler için
        a_data = np.random.randn(5, 5).astype(np.float32)
        b_data = np.random.randn(5, 5).astype(np.float32)
        a_az, b_az = Tensor(a_data, requires_grad=True), Tensor(b_data, requires_grad=True)
        a_th, b_th = torch.tensor(a_data, requires_grad=True), torch.tensor(b_data, requires_grad=True)
        op_map = {
            "add": (a_az + b_az, a_th + b_th), "sub": (a_az - b_az, a_th - b_th),
            "mul": (a_az * b_az, a_th * b_th), "div": (a_az / b_az, a_th / b_th),
            "pow": (a_az ** 3, a_th ** 3), "sum": (a_az.sum(), a_th.sum()),
            "mean": (a_az.mean(), a_th.mean()), "relu": (a_az.relu(), a_th.relu()),
            "sigmoid": (a_az.sigmoid(), a_th.sigmoid()), "tanh": (a_az.tanh(), a_th.tanh()),
        }
        out_az_op, out_th_op = op_map[op]
        out_az, out_th = out_az_op.sum(), out_th_op.sum()

    out_az.backward()
    out_th.backward()

    assert np.allclose(a_az.grad, a_th.grad.numpy(), atol=1e-5), f"Gradient mismatch for op '{op}' on tensor 'a'"
    if op in ["add", "sub", "mul", "div", "dot"]:
        assert np.allclose(b_az.grad, b_th.grad.numpy(), atol=1e-5), f"Gradient mismatch for op '{op}' on tensor 'b'"
    if op == "conv2d":
        assert np.allclose(b_az.grad, b_th.grad.numpy(), atol=1e-5), f"Gradient mismatch for op '{op}' on weights"
        assert np.allclose(c_az.grad, c_th.grad.numpy(), atol=1e-5), f"Gradient mismatch for op '{op}' on bias"