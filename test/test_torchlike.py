# Test ability to type-check user defined classes which have a "torch-like" interface
# The required interface is defined as the protocol TensorLike in tensor_details.py

from __future__ import annotations
import pytest
import torch
from torch import rand

from torchtyping import TensorType, TensorTypeMixin
from typeguard import typechecked


# New class that supports the tensor-like interface
class MyTensor:
    def __init__(self, tensor: torch.Tensor = torch.zeros(2, 3)):
        self.tensor = tensor
        self.dtype = self.tensor.dtype
        self.layout = "something special"
        self.names = self.tensor.names
        self.shape = self.tensor.shape

    def is_floating_point(self) -> bool:
        return self.dtype == torch.float32

    # Add tensors and take the mean over the last dimension
    # Output drops the last dimension
    def __add__(self, o: torch.Tensor) -> MyTensor:
        res = self.tensor + o
        res_reduced = torch.mean(res, -1)
        res_myt = MyTensor(res_reduced)
        return res_myt


# Create a type corresponding to the new class
class MyTensorType(MyTensor, TensorTypeMixin):
    base_cls = MyTensor


def test_my_tensor1():
    @typechecked
    def func(x: MyTensorType["x", "y"], y: TensorType["x", "y"]) -> MyTensorType["x"]:
        return x + y

    @typechecked
    def bad_func_spec(x: MyTensorType["x", "y"], y: TensorType["x", "y"]) -> MyTensorType["x", "y"]:
        return x + y

    my_t: MyTensor = MyTensor()
    func(my_t, rand((2, 3)))

    # Incorrect input dimensions for x
    with pytest.raises(TypeError):
        func(MyTensor(rand(1)), rand((2, 3)))

    # Incorrect input dimensions for y
    with pytest.raises(TypeError):
        func(my_t, rand(1))

    # Incorrect spec for return dimensions
    with pytest.raises(TypeError):
        bad_func_spec(my_t, rand((2, 3)))
