import pytest
from torch import rand
from torchtyping import TensorType
from typeguard import typechecked


x = y = None


def test_single():
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


def test_multiple():
    # Fun fact, this "wrong" func0 is actually a mistype of func1, that torchtyping
    # caught for me when I ran the tests!
    @typechecked
    def func0(x: TensorType["x"], y: TensorType["y"]) -> TensorType["x", "y"]:
        return x.unsqueeze(0) + y.unsqueeze(1)

    @typechecked
    def func1(x: TensorType["x"], y: TensorType["y"]) -> TensorType["x", "y"]:
        return x.unsqueeze(1) + y.unsqueeze(0)

    @typechecked
    def func2(x: TensorType["x", "x"]):
        pass

    @typechecked
    def func3(x: TensorType["x", "x", "x"]):
        pass

    @typechecked
    def func4(x: TensorType["x"], y: TensorType["x", "y"]):
        pass

    @typechecked
    def func5(x: TensorType["x", "y"], y: TensorType["y", "x"]):
        pass

    @typechecked
    def func6(x: TensorType["x"], y: TensorType["y"]) -> TensorType["x", "y"]:
        assert not (x.shape == (2,) and y.shape == (3,))
        return rand(2, 3)

    func0(rand(2), rand(2))  # can't catch this
    with pytest.raises(TypeError):
        func0(rand(2), rand(3))
    with pytest.raises(TypeError):
        func0(rand(10), rand(0))

    func1(rand(2), rand(2))
    func1(rand(2), rand(3))
    func1(rand(10), rand(0))

    func2(rand(0, 0))
    func2(rand(2, 2))
    func2(rand(9, 9))
    with pytest.raises(TypeError):
        func2(rand(0, 4))
        func2(rand(1, 4))
        func2(rand(3, 4))

    func3(rand(0, 0, 0))
    func3(rand(2, 2, 2))
    func3(rand(9, 9, 9))
    with pytest.raises(TypeError):
        func3(rand(0, 4, 4))
        func3(rand(1, 4, 4))
        func3(rand(3, 3, 4))

    func4(rand(3), rand(3, 4))
    with pytest.raises(TypeError):
        func4(rand(3), rand(4, 3))

    func5(rand(2, 3), rand(3, 2))
    func5(rand(0, 5), rand(5, 0))
    func5(rand(2, 2), rand(2, 2))
    with pytest.raises(TypeError):
        func5(rand(2, 3), rand(2, 3))
        func5(rand(2, 3), rand(2, 2))

    with pytest.raises(TypeError):
        func6(rand(5), rand(3))
