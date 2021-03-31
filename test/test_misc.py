import pytest
import torch
from torchtyping import TensorType
from typeguard import typechecked


dim1 = dim2 = channel = None


def test_non_tensor():
    class Tensor:
        shape = torch.Size([2, 2])
        dtype = torch.float32
        layout = torch.strided

    args = (None, 4, 3.0, 3.2, "Tensor", Tensor, Tensor())

    @typechecked
    def accepts_tensor1(x: TensorType):
        pass

    @typechecked
    def accepts_tensor2(x: TensorType[2, 2]):
        pass

    @typechecked
    def accepts_tensor3(x: TensorType[...]):
        pass

    @typechecked
    def accepts_tensor4(x: TensorType[float]):
        pass

    @typechecked
    def accepts_tensor5(x: TensorType[..., float]):
        pass

    @typechecked
    def accepts_tensor6(x: TensorType[2, int]):
        pass

    @typechecked
    def accepts_tensor7(x: TensorType[torch.strided]):
        pass

    @typechecked
    def accepts_tensor8(x: TensorType[2, float, torch.sparse_coo]):
        pass

    for func in (
        accepts_tensor1,
        accepts_tensor2,
        accepts_tensor3,
        accepts_tensor4,
        accepts_tensor5,
        accepts_tensor6,
        accepts_tensor7,
        accepts_tensor8,
    ):
        for arg in args:
            with pytest.raises(TypeError):
                func(arg)

    @typechecked
    def accepts_tensors1(x: TensorType, y: TensorType):
        pass

    @typechecked
    def accepts_tensors2(x: TensorType[2, 2], y: TensorType):
        pass

    @typechecked
    def accepts_tensors3(x: TensorType[...], y: TensorType):
        pass

    @typechecked
    def accepts_tensors4(x: TensorType[float], y: TensorType):
        pass

    @typechecked
    def accepts_tensors5(x: TensorType[..., float], y: TensorType):
        pass

    @typechecked
    def accepts_tensors6(x: TensorType[2, int], y: TensorType):
        pass

    @typechecked
    def accepts_tensors7(x: TensorType[torch.strided], y: TensorType):
        pass

    @typechecked
    def accepts_tensors8(x: TensorType[torch.sparse_coo, float, 2], y: TensorType):
        pass

    for func in (
        accepts_tensors1,
        accepts_tensors2,
        accepts_tensors3,
        accepts_tensors4,
        accepts_tensors5,
        accepts_tensors6,
        accepts_tensors7,
        accepts_tensors8,
    ):
        for arg1 in args:
            for arg2 in args:
                with pytest.raises(TypeError):
                    func(arg1, arg2)


def test_multiple_ellipsis():
    @typechecked
    def func(
        x: TensorType["dim1":..., "dim2":...], y: TensorType["dim2":...]
    ) -> TensorType["dim1":...]:
        sum_dims = [x - 1 for x in range(y.dim())]
        return (x * y).sum(dim=sum_dims)

    func(torch.rand(1, 2), torch.rand(2))
    func(torch.rand(3, 4, 5, 9), torch.rand(5, 9))
    with pytest.raises(TypeError):
        func(torch.rand(1), torch.rand(2))
    with pytest.raises(TypeError):
        func(torch.rand(1, 3, 5), torch.rand(3))
    with pytest.raises(TypeError):
        func(torch.rand(1, 4), torch.rand(1, 1, 4))


def test_nested_types():
    @typechecked
    def func(x: tuple[TensorType[3, "channel", 4], TensorType["channel"]]):
        pass

    func((torch.rand(3, 1, 4), torch.rand(1)))
    func((torch.rand(3, 5, 4), torch.rand(5)))
    with pytest.raises(TypeError):
        func((torch.rand(3, 1, 4), torch.rand(2)))
