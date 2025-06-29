# ========== YENİ DOSYA: tests\azuraforge_core\test_tensor.py ==========
import pytest
import numpy as np

# Test edilecek paketi import et
from azuraforge_core import Tensor

def test_tensor_creation_and_defaults():
    """Tensor nesnesinin doğru şekilde ve varsayılan değerlerle oluşturulduğunu test eder."""
    t = Tensor([1, 2, 3])
    assert isinstance(t.data, np.ndarray)
    assert t.requires_grad is False
    assert t.grad is None

def test_tensor_requires_grad():
    """`requires_grad=True` olduğunda gradyan dizisinin oluşturulduğunu test eder."""
    t = Tensor([1, 2], requires_grad=True)
    assert t.requires_grad is True
    assert isinstance(t.grad, np.ndarray)
    assert np.array_equal(t.grad, np.array([0.0, 0.0]))

def test_addition_backward():
    """Basit toplama işlemi için geri yayılımın doğru çalıştığını test eder."""
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor(5, requires_grad=True)
    
    # c = a.sum() + b  ->  dc/da = [1, 1, 1], dc/db = 1
    c = a.sum() + b
    
    c.backward()

    assert a.grad is not None
    assert b.grad is not None
    assert np.array_equal(a.grad, [1, 1, 1])
    assert b.grad == 1.0

def test_multiplication_backward():
    """Basit çarpma işlemi için geri yayılımın doğru çalıştığını test eder."""
    x = Tensor(2.0, requires_grad=True)
    y = Tensor(3.0, requires_grad=True)
    
    z = x * y
    
    z.backward() # dz/dx = y = 3,  dz/dy = x = 2

    assert x.grad == 3.0
    assert y.grad == 2.0

def test_chained_rule_backward():
    """Zincir kuralının birden çok işlemde doğru çalıştığını test eder."""
    x = Tensor(2.0, requires_grad=True)
    y = Tensor(3.0, requires_grad=True)

    z = x * y  # dz/dx = y, dz/dy = x
    q = z + x  # dq/dz = 1, dq/dx = 1
    
    # Zincir Kuralı:
    # dq/dx = (dq/dz * dz/dx) + dq/dx = (1 * y) + 1 = 3 + 1 = 4
    # dq/dy = (dq/dz * dz/dy) = 1 * x = 2
    q.backward()

    assert x.grad == 4.0
    assert y.grad == 2.0

def test_dot_product_backward():
    """Matris çarpımı için geri yayılımı test eder."""
    a_data = np.random.randn(2, 3)
    b_data = np.random.randn(3, 4)
    a = Tensor(a_data, requires_grad=True)
    b = Tensor(b_data, requires_grad=True)
    
    c = a.dot(b)
    
    # Gradyanı 1'lerden oluşan bir matrisle başlat
    c.backward(np.ones_like(c.data))
    
    # Manuel olarak hesaplanan gradyanlar
    grad_a_manual = np.ones_like(c.data) @ b_data.T
    grad_b_manual = a_data.T @ np.ones_like(c.data)
    
    assert np.allclose(a.grad, grad_a_manual)
    assert np.allclose(b.grad, grad_b_manual)


def test_relu_backward():
    """ReLU aktivasyonu için geri yayılımı test eder."""
    a = Tensor([-1, 0, 5], requires_grad=True)
    r = a.relu()
    r.backward(np.array([10, 20, 30]))

    # Gradyan sadece pozitif değerler için akar (self.data > 0)
    assert np.array_equal(a.grad, [0, 0, 30])