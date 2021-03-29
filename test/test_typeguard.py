import pytest
from torch import rand
from torchtyping import TensorType
from typeguard import typechecked


@pytest.mark.patch_typeguard
def test_same():
    @typechecked
    def func1(x: TensorType["x"], y: TensorType["x"]):
        pass

    @typechecked
    def func2(x: TensorType["x"], y: TensorType["x"]) -> TensorType["x"]:
        return x + y

    @typechecked
    def func3(x: TensorType["x"], y: TensorType["x"]) -> TensorType["x", "x"]:
        return x + y

    @typechecked
    def func4(x: TensorType["x"], y: TensorType["x"]) -> TensorType["x", "x"]:
        return x.unsqueeze(0) + y.unsqueeze(1)

    @typechecked
    def func5(x: TensorType["x"], y: TensorType["x"]) -> TensorType["x", "y"]:
        return x

    @typechecked
    def func6(x: TensorType["x"], y: TensorType["x"]) -> TensorType["y", "x"]:
        return x

    @typechecked
    def func7(x: TensorType["x"]) -> TensorType["x"]:
        assert x.shape != (1,)
        return rand((1,))

    func1(rand(2), rand(2))
    func2(rand(2), rand(2))
    with pytest.raises(TypeError):
        func3(rand(2), rand(2))
    func4(rand(2), rand(2))
    with pytest.raises(TypeError):
        func5(rand(2), rand(2))
    with pytest.raises(TypeError):
        func6(rand(2), rand(2))
    with pytest.raises(TypeError):
        func7(rand(3))
