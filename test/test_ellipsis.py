import pytest
import torch
from torchtyping import TensorType
from typeguard import typechecked


dim1 = dim2 = dim3 = channel = None


def test_basic_ellipsis():
    @typechecked
    def func(x: TensorType["dim1":..., "dim2", "dim3"], y: TensorType["dim2", "dim3"]):
        pass

    func(torch.rand(2, 2), torch.rand(2, 2))
    func(torch.rand(2, 3), torch.rand(2, 3))
    func(torch.rand(1, 4, 2, 3), torch.rand(2, 3))
    func(torch.rand(2, 3, 2, 3), torch.rand(2, 3))

    with pytest.raises(TypeError):
        func(torch.rand(2, 3), torch.rand(3, 2))
    with pytest.raises(TypeError):
        func(torch.rand(1, 4, 2, 3), torch.rand(3, 2))
    with pytest.raises(TypeError):
        func(torch.rand(2, 3, 2, 3), torch.rand(3, 2))


def test_zero_size_ellipsis1():
    @typechecked
    def func(
        x: TensorType["dim1":..., "dim2", "dim3"],
        y: TensorType["dim1":..., "dim2", "dim3"],
    ):
        pass

    with pytest.raises(TypeError):
        func(torch.rand(2, 2), torch.rand(2, 2, 2))
    with pytest.raises(TypeError):
        func(torch.rand(2, 2, 2), torch.rand(2, 2))


def test_zero_size_ellipsis2():
    @typechecked
    def func(
        x: TensorType["dim1":..., "dim2", "dim3"],
        y: TensorType["dim1":..., "dim2", "dim3"],
    ):
        pass

    with pytest.raises(TypeError):
        func(torch.rand(2, 3), torch.rand(2, 2, 3))
    with pytest.raises(TypeError):
        func(torch.rand(2, 2, 3), torch.rand(2, 3))
    with pytest.raises(TypeError):
        func(torch.rand(2, 2), torch.rand(2, 2, 2))
    with pytest.raises(TypeError):
        func(torch.rand(2, 2, 2), torch.rand(2, 2))


def test_multiple_ellipsis1():
    @typechecked
    def func(
        x: TensorType["dim1":..., "dim2":...], y: TensorType["dim2":...]
    ) -> TensorType["dim1":...]:
        sum_dims = [-i - 1 for i in range(y.dim())]
        return (x * y).sum(dim=sum_dims)

    func(torch.rand(1, 2), torch.rand(2))
    func(torch.rand(3, 4, 5, 9), torch.rand(5, 9))
    func(torch.rand(3, 4, 11, 5, 9), torch.rand(5, 9))
    func(torch.rand(3, 4, 11, 5, 9), torch.rand(11, 5, 9))
    with pytest.raises(TypeError):
        func(torch.rand(1), torch.rand(2))
    with pytest.raises(TypeError):
        func(torch.rand(1, 3, 5), torch.rand(3))
    with pytest.raises(TypeError):
        func(torch.rand(1, 4), torch.rand(1, 1, 4))


def test_multiple_ellipsis2():
    @typechecked
    def func(
        x: TensorType["dim2":...], y: TensorType["dim1":..., "dim2":...]
    ) -> TensorType["dim1":...]:
        sum_dims = [-i - 1 for i in range(x.dim())]
        return (x * y).sum(dim=sum_dims)

    with pytest.raises(TypeError):
        func(torch.rand(1, 1, 4), torch.rand(1, 4))


def test_multiple_ellipsis3():
    @typechecked
    def func(
        x: TensorType["dim1":..., "dim2":..., "dim3":...],
        y: TensorType["dim2":..., "dim3":...],
        z: TensorType["dim2":...],
    ) -> TensorType["dim1":...]:
        num2 = y.dim() - z.dim()
        num3 = z.dim()
        for _ in range(num2):
            z = z.unsqueeze(-1)
        y = y * z
        x = x + y
        for _ in range(num2 + num3):
            x = x.sum(dim=-1)
        return x

    func(torch.rand(1, 2, 3), torch.rand(2, 3), torch.rand(2))
    func(torch.rand(3, 5, 6, 7, 8, 0), torch.rand(5, 6, 7, 8, 0), torch.rand(5, 6, 7))
    func(torch.rand(3, 5, 6, 7, 8, 9), torch.rand(5, 6, 7, 8, 9), torch.rand(5, 6, 7))


def test_repeat_ellipsis1():
    @typechecked
    def func(x: TensorType["dim1":..., "dim1":...], y: TensorType["dim1":...]):
        pass

    func(torch.rand(3, 4, 3, 4), torch.rand(3, 4))
    func(torch.rand(5, 5), torch.rand(5))
    with pytest.raises(TypeError):
        func(torch.rand(7, 9), torch.rand(7))
    with pytest.raises(TypeError):
        func(torch.rand(7, 4, 9, 4), torch.rand(7, 4))
    with pytest.raises(TypeError):
        func(torch.rand(7, 9), torch.rand(9))
    with pytest.raises(TypeError):
        func(torch.rand(3, 7, 3, 9), torch.rand(3, 9))
    with pytest.raises(TypeError):
        func(torch.rand(7, 3, 3, 9), torch.rand(3, 9))
    with pytest.raises(TypeError):
        func(torch.rand(7, 7), torch.rand(3))


def test_repeat_ellipsis2():
    @typechecked
    def func(
        x: TensorType["dim1":..., "dim1":...],
        y: TensorType["dim1":..., "dim2":...],
        z: TensorType["dim2":...],
    ):
        pass

    func(torch.rand(4, 4), torch.rand(4, 5), torch.rand(5))
    func(torch.rand(3, 4, 3, 4), torch.rand(3, 4, 5), torch.rand(5))
    func(torch.rand(2, 3, 4, 2, 3, 4), torch.rand(2, 3, 4, 5, 6), torch.rand(5, 6))
    with pytest.raises(TypeError):
        func(torch.rand(2, 3, 4, 2, 3), torch.rand(2, 3, 4, 5, 6), torch.rand(5, 6))
    with pytest.raises(TypeError):
        func(torch.rand(2, 3, 4, 2, 3), torch.rand(2, 3, 4, 6), torch.rand(3, 4))


def test_ambiguous_ellipsis():
    @typechecked
    def func1(x: TensorType["dim1":..., "dim2":...]):
        pass

    with pytest.raises(TypeError):
        func1(torch.rand(2, 2))

    @typechecked
    def func2(x: TensorType["dim1":..., "dim1":...]):
        pass

    with pytest.raises(TypeError):
        func2(torch.rand(2, 2))
