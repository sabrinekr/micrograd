"""Test value class functionalities."""
import torch

from micrograd.value import Value


def test_value_operations() -> None:
    """
    Check the value class operations.
    """
    # inputs x1_value, x2_value
    x1_value = Value(2.0, label="x1")
    x2_value = Value(0.0, label="x2")
    # inputs w1_value, w2_value
    w1_value = Value(-3.0, label="w1")
    w2_value = Value(1.0, label="w2")
    # bias of the neuron
    b_value = Value(6.0, label="b")

    x1w1_value = x1_value*w1_value
    x2w2_value = x2_value*w2_value
    x1w1x2w2_value = x1w1_value + x2w2_value
    n_value = x1w1x2w2_value + b_value
    o_value = n_value.tanh() 
    o_value.backward()

    # inputs x1_torch, x2_torch
    x1_torch = torch.tensor(2.0, requires_grad=True)
    x2_torch = torch.tensor(0.0, requires_grad=True)
    # inputs w1_torch, w2_torch
    w1_torch = torch.tensor(-3.0, requires_grad=True)
    w2_torch = torch.tensor(1.0, requires_grad=True)
    # bias of the neuron
    b_torch = torch.tensor(6.0, requires_grad=True)

    x1w1_torch = x1_torch * w1_torch 
    x2w2_torch = x2_torch * w2_torch
    x1w1x2w2_torch = x1w1_torch + x2w2_torch
    n_torch = x1w1x2w2_torch + b_torch
    o_torch = torch.tanh(n_torch)
    o_torch.backward()

    assert x1_torch.grad == x1_value.grad
    assert x2_torch.grad == x2_value.grad
    assert o_value.data == o_torch.item()

def test_more_value_operations() -> None:
    """
    Check the value class operations.
    """
    x1_value = Value(2.0)
    x2_value = Value(0.0)
    x3_value = Value(-3.0)
    x4_value = Value(1.0)
    b_value = (x1_value - x2_value)**2 /x3_value
    o_value = (b_value*x4_value).relu() 
    o_value.backward()

    x1_torch = torch.tensor(2.0, requires_grad=True)
    x2_torch = torch.tensor(0.0, requires_grad=True)
    x3_torch = torch.tensor(-3.0, requires_grad=True)
    x4_torch = torch.tensor(1.0, requires_grad=True)
    b_torch = (x1_torch - x2_torch)**2 / x3_torch
    o_torch = (b_torch * x4_torch).relu()
    o_torch.backward()

    assert x1_torch.grad == x1_value.grad
    assert x2_torch.grad == x2_value.grad
    assert o_value.data == o_torch.item()