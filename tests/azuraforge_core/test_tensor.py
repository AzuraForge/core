import pytest
import numpy as np

# PyTorch'u sadece gradyan doğruluğunu kontrol etmek için bir "orakl" (doğruluk kaynağı) olarak kullanacağız.
# Projenin çalışma zamanı bağımlılığı olmayacak. Test ortamına `pip install torch` ile kurulması gerekir.
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

def test_tensor_requires_grad():
    t = Tensor([1, 2], requires_grad=True)
    assert t.requires_grad is True
    assert isinstance(t.grad, xp.ndarray)
    assert np.allclose(t.grad, xp.zeros_like(t.data))

# --- Gradyan Doğruluk Testleri (PyTorch ile Karşılaştırmalı) ---

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed, skipping gradient checks.")
@pytest.mark.parametrize("op", [
    "add", "sub", "mul", "div", "pow", "dot", "sum", "mean", "relu", "sigmoid", "tanh"
])
def test_gradient_correctness_vs_pytorch(op):
    """Farklı operasyonlar için gradyanların PyTorch sonuçlarıyla eşleştiğini doğrular."""
    
    # Bizim Tensor'lerimiz
    a_data = np.random.randn(2, 3).astype(np.float32)
    b_data = np.random.randn(3, 4).astype(np.float32)
    c_data = np.random.randn(2, 3).astype(np.float32)

    a_az = Tensor(a_data, requires_grad=True)
    b_az = Tensor(b_data, requires_grad=True)
    c_az = Tensor(c_data, requires_grad=True)

    # PyTorch Tensor'leri
    a_th = torch.tensor(a_data, requires_grad=True)
    b_th = torch.tensor(b_data, requires_grad=True)
    c_th = torch.tensor(c_data, requires_grad=True)

    # İşlemleri yap
    if op == "add":
        out_az = (a_az + c_az).sum()
        out_th = (a_th + c_th).sum()
    elif op == "sub":
        out_az = (a_az - c_az).sum()
        out_th = (a_th - c_th).sum()
    elif op == "mul":
        out_az = (a_az * c_az).sum()
        out_th = (a_th * c_th).sum()
    elif op == "div":
        out_az = (a_az / c_az).sum()
        out_th = (a_th / c_th).sum()
    elif op == "pow":
        out_az = (a_az ** 3).sum()
        out_th = (a_th ** 3).sum()
    elif op == "dot":
        out_az = a_az.dot(b_az)
        out_th = a_th @ b_th
    elif op == "sum":
        out_az = a_az.sum()
        out_th = a_th.sum()
    elif op == "mean":
        out_az = a_az.mean()
        out_th = a_th.mean()
    elif op == "relu":
        out_az = a_az.relu().sum()
        out_th = a_th.relu().sum()
    elif op == "sigmoid":
        out_az = a_az.sigmoid().sum()
        out_th = a_th.sigmoid().sum()
    elif op == "tanh":
        out_az = a_az.tanh().sum()
        out_th = a_th.tanh().sum()
    else:
        return

    # Geriye yayılım
    out_az.backward()
    out_th.backward(torch.ones_like(out_th))

    # Gradyanları karşılaştır
    assert a_az.grad is not None and a_th.grad is not None
    assert np.allclose(a_az.grad, a_th.grad.numpy(), atol=1e-6), f"Gradient mismatch for op '{op}' on tensor 'a'"
    
    if op == "dot":
      assert b_az.grad is not None and b_th.grad is not None
      assert np.allclose(b_az.grad, b_th.grad.numpy(), atol=1e-6), f"Gradient mismatch for op '{op}' on tensor 'b'"
    elif op in ["add", "sub", "mul", "div"]:
      assert c_az.grad is not None and c_th.grad is not None
      assert np.allclose(c_az.grad, c_th.grad.numpy(), atol=1e-6), f"Gradient mismatch for op '{op}' on tensor 'c'"

# --- Karmaşık Senaryo Testi ---

def test_chained_rule_and_broadcasting():
    """Zincir kuralı ve broadcasting içeren daha karmaşık bir senaryoyu test eder."""
    a_data = np.random.randn(2, 3).astype(np.float32)
    b_data = np.array([10, 20, 30]).astype(np.float32) # Broadcasting için
    
    a_az = Tensor(a_data, requires_grad=True)
    b_az = Tensor(b_data, requires_grad=True)

    # PyTorch versiyonu
    a_th = torch.tensor(a_data, requires_grad=True)
    b_th = torch.tensor(b_data, requires_grad=True)

    # AzuraForge: c = (a * b).relu() -> d = c.mean()
    c_az = (a_az * b_az).relu()
    d_az = c_az.mean()
    d_az.backward()

    # PyTorch: c = (a * b).relu() -> d = c.mean()
    c_th = (a_th * b_th).relu()
    d_th = c_th.mean()
    d_th.backward()

    # Gradyanları karşılaştır
    assert np.allclose(a_az.grad, a_th.grad.numpy(), atol=1e-6)
    assert np.allclose(b_az.grad, b_th.grad.numpy(), atol=1e-6)