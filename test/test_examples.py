import pytest

from torch import rand, sparse_coo, tensor
from torchtyping import NamedTensorType, TensorType
from typeguard import typechecked


@pytest.mark.torchtyping_patch_typeguard
def test_example0():
    @typechecked
    def batch_outer_product(
        x: TensorType["batch", "x_channels"], y: TensorType["batch", "y_channels"]
    ) -> TensorType["batch", "x_channels", "y_channels"]:

        return x.unsqueeze(-1) * y.unsqueeze(-2)

    batch_outer_product(rand(2, 3), rand(2, 4))
    batch_outer_product(rand(5, 2), rand(5, 2))
    with pytest.raises(TypeError):
        batch_outer_product(rand(3, 2), rand(2, 3))
    with pytest.raises(TypeError):
        batch_outer_product(rand(1, 2, 3), rand(2, 3))


@pytest.mark.torchtyping_patch_typeguard
def test_example1():
    @typechecked
    def func(x: TensorType["batch"], y: TensorType["batch"]) -> TensorType["batch"]:
        return x + y

    func(rand(3), rand(3))
    with pytest.raises(TypeError):
        func(rand(3), rand(1))


@pytest.mark.torchtyping_patch_typeguard
def test_example2():
    @typechecked
    def func(x: TensorType["batch", 5], y: TensorType["batch", 3]):
        pass

    func(rand(3, 5), rand(3, 3))
    func(rand(7, 5), rand(7, 3))
    with pytest.raises(TypeError):
        func(rand(4, 5), rand(3, 5))
    with pytest.raises(TypeError):
        func(rand(1, 3, 5), rand(3, 5))
    with pytest.raises(TypeError):
        func(rand(3, 4), rand(3, 3))
    with pytest.raises(TypeError):
        func(rand(1, 3, 5), rand(1, 3, 3))


def test_example3():
    @typechecked
    def func(x: TensorType[2, -1, -1]):
        pass

    func(rand(2, 1, 1))
    func(rand(2, 2, 1))
    func(rand(2, 10, 1))
    func(rand(2, 1, 10))
    with pytest.raises(TypeError):
        func(rand(1, 2, 1, 1))
    with pytest.raises(TypeError):
        func(rand(2, 1))
    with pytest.raises(TypeError):
        func(
            rand(
                2,
            )
        )
    with pytest.raises(TypeError):
        func(rand(4, 2, 2))


def test_example4():
    @typechecked
    def func(x: TensorType[..., 2, 3]):
        pass

    func(rand(2, 3))
    func(rand(1, 2, 3))
    func(rand(2, 2, 3))
    func(rand(3, 3, 5, 2, 3))
    with pytest.raises(TypeError):
        func(rand(1, 3))
    with pytest.raises(TypeError):
        func(rand(2, 1))
    with pytest.raises(TypeError):
        func(rand(1, 1, 3))
    with pytest.raises(TypeError):
        func(rand(2, 3, 3))
    with pytest.raises(TypeError):
        func(rand(3))
    with pytest.raises(TypeError):
        func(rand(2))


@pytest.mark.torchtyping_patch_typeguard
def test_example5():
    @typechecked
    def func(x: TensorType[..., 2, "channels"], y: TensorType[..., "channels"]):
        pass

    func(rand(1, 2, 3), rand(3))
    func(rand(1, 2, 3), rand(1, 3))
    func(rand(2, 3), rand(2, 3))
    with pytest.raises(TypeError):
        func(rand(2, 2, 2, 2), rand(2, 4))
    with pytest.raises(TypeError):
        func(rand(3, 2), rand(2))
    with pytest.raises(TypeError):
        func(rand(5, 2, 1), rand(2))


@pytest.mark.torchtyping_patch_typeguard
def test_example6():
    @typechecked
    def func(
        x: TensorType["batch":..., "channels_x"],
        y: TensorType["batch":..., "channels_y"],
    ):
        pass

    func(rand(3, 3, 3), rand(3, 3, 4))
    func(rand(1, 5, 6, 7), rand(1, 5, 6, 1))
    with pytest.raises(TypeError):
        func(rand(2, 2, 2), rand(2, 1, 2))
    with pytest.raises(TypeError):
        func(rand(4, 2, 2), rand(2, 2, 2))
    with pytest.raises(TypeError):
        func(rand(2, 2, 2), rand(2, 1, 4))
    with pytest.raises(TypeError):
        func(rand(2, 2), rand(2, 1, 2))
    with pytest.raises(TypeError):
        func(rand(2, 2), rand(1, 2, 2))
    with pytest.raises(TypeError):
        func(rand(4, 2), rand(3, 2))


def test_example7():
    @typechecked
    def func(x: TensorType[3, 4]) -> TensorType[()]:
        return rand(())

    func(rand(3, 4))
    with pytest.raises(TypeError):
        func(rand(2, 4))

    @typechecked
    def func2(x: TensorType[3, 4]) -> TensorType[()]:
        return rand((1,))

    with pytest.raises(TypeError):
        func2(rand(3, 4))
    with pytest.raises(TypeError):
        func2(rand(2, 4))


def test_example8():
    @typechecked
    def func(x: TensorType[float]):
        pass

    func(rand(2, 3))
    func(rand(1))
    func(rand(()))
    with pytest.raises(TypeError):
        func(tensor(1))
    with pytest.raises(TypeError):
        func(tensor([1, 2]))
    with pytest.raises(TypeError):
        func(tensor([[1, 1], [2, 2]]))
    with pytest.raises(TypeError):
        func(tensor(True))


def test_example9():
    @typechecked
    def func(x: TensorType[3, 4][float]):
        pass

    func(rand(3, 4))
    with pytest.raises(TypeError):
        func(rand(3, 4).long())
    with pytest.raises(TypeError):
        func(rand(2, 3))


def test_example10():
    @typechecked
    def func(x: NamedTensorType["a":3, "b"]):
        pass

    func(rand(3, 4, names=("a", "b")))
    with pytest.raises(TypeError):
        func(rand(3, 4), names=("a", "c"))
    with pytest.raises(TypeError):
        func(rand(3, 3, 3), names=(None, "a", "b"))
    with pytest.raises(TypeError):
        func(rand(3, 3, 3), names=("a", None, "b"))
    with pytest.raises(TypeError):
        func(rand(3, 3, 3), names=("a", "b", None))
    with pytest.raises(TypeError):
        func(rand(3, 4, names=(None, "b")))
    with pytest.raises(TypeError):
        func(rand(3, 4, names=("a", None)))
    with pytest.raises(TypeError):
        func(rand(3, 4))
    with pytest.raises(TypeError):
        func(rand(3, 4).long())
    with pytest.raises(TypeError):
        func(rand(3))


def test_example11():
    @typechecked
    def func(x: TensorType[sparse_coo]):
        pass

    func(rand(3, 4).to_sparse())
    with pytest.raises(TypeError):
        func(rand(3, 4))
    with pytest.raises(TypeError):
        func(rand(3, 4).long())
