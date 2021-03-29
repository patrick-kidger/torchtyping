import pytest
import torch
from torchtyping import FloatTensorType, NamedTensorType, NamedFloatTensorType
import typeguard


def test_float_tensor():
    @typeguard.typechecked
    def func1(x: FloatTensorType):
        pass

    @typeguard.typechecked
    def func2(x: FloatTensorType[2, 2]):
        pass

    @typeguard.typechecked
    def func3(x: FloatTensorType[float]):
        pass

    @typeguard.typechecked
    def func4(x: FloatTensorType[bool]):
        pass

    @typeguard.typechecked
    def func5(x: FloatTensorType["a":2, 2][float]):
        pass

    @typeguard.typechecked
    def func6(x: FloatTensorType[2, "b":2][torch.sparse_coo]):
        pass

    x = torch.rand(2, 2)
    y = torch.rand(1)
    z = torch.tensor([[0, 1], [2, 3]])
    w = torch.rand(4).to_sparse()
    a = torch.rand(2, 2).to_sparse()
    b = torch.tensor([[0, 1], [2, 3]]).to_sparse()

    func1(x)
    func1(y)
    with pytest.raises(TypeError):
        func1(z)
    func1(w)
    func1(a)
    with pytest.raises(TypeError):
        func1(b)

    func2(x)
    with pytest.raises(TypeError):
        func2(y)
    with pytest.raises(TypeError):
        func2(z)
    with pytest.raises(TypeError):
        func2(w)
    func2(a)
    with pytest.raises(TypeError):
        func2(b)

    func3(x)
    func3(y)
    with pytest.raises(TypeError):
        func3(z)
    func3(w)
    func3(a)
    with pytest.raises(TypeError):
        func3(b)

    with pytest.raises(TypeError):
        func4(x)
    with pytest.raises(TypeError):
        func4(y)
    with pytest.raises(TypeError):
        func4(z)
    with pytest.raises(TypeError):
        func4(w)
    with pytest.raises(TypeError):
        func4(a)
    with pytest.raises(TypeError):
        func4(b)

    func5(x)
    with pytest.raises(TypeError):
        func5(y)
    with pytest.raises(TypeError):
        func5(z)
    with pytest.raises(TypeError):
        func5(w)
    func5(a)
    with pytest.raises(TypeError):
        func5(b)

    with pytest.raises(TypeError):
        func6(x)
    with pytest.raises(TypeError):
        func6(y)
    with pytest.raises(TypeError):
        func6(z)
    with pytest.raises(TypeError):
        func6(w)
    func6(a)
    with pytest.raises(TypeError):
        func6(b)


def test_named_tensor():
    @typeguard.typechecked
    def _named_a_dim_checker(x: NamedTensorType["a"]):
        pass

    @typeguard.typechecked
    def _named_ab_dim_checker(x: NamedTensorType["a", "b"]):
        pass

    @typeguard.typechecked
    def _named_abc_dim_checker(x: NamedTensorType["a", "b", "c"]):
        pass

    @typeguard.typechecked
    def _named_cb_dim_checker(x: NamedTensorType["c", "b"]):
        pass

    @typeguard.typechecked
    def _named_am1_dim_checker(x: NamedTensorType["a", -1]):
        pass

    @typeguard.typechecked
    def _named_m1b_dim_checker(x: NamedTensorType[-1, "b"]):
        pass

    @typeguard.typechecked
    def _named_abm1_dim_checker(x: NamedTensorType["a", "b", -1]):
        pass

    @typeguard.typechecked
    def _named_m1bm1_dim_checker(x: NamedTensorType[-1, "b", -1]):
        pass

    x = torch.rand(3, 4)
    named_x = torch.rand(3, 4, names=("a", "b"))

    with pytest.raises(TypeError):
        _named_ab_dim_checker(x)
    with pytest.raises(TypeError):
        _named_cb_dim_checker(x)
    with pytest.raises(TypeError):
        _named_am1_dim_checker(x)
    with pytest.raises(TypeError):
        _named_m1b_dim_checker(x)
    with pytest.raises(TypeError):
        _named_a_dim_checker(x)
    with pytest.raises(TypeError):
        _named_abc_dim_checker(x)
    with pytest.raises(TypeError):
        _named_abm1_dim_checker(x)
    with pytest.raises(TypeError):
        _named_m1bm1_dim_checker(x)

    _named_ab_dim_checker(named_x)
    _named_am1_dim_checker(named_x)
    _named_m1b_dim_checker(named_x)
    with pytest.raises(TypeError):
        _named_a_dim_checker(named_x)
    with pytest.raises(TypeError):
        _named_abc_dim_checker(named_x)
    with pytest.raises(TypeError):
        _named_cb_dim_checker(named_x)
    with pytest.raises(TypeError):
        _named_abm1_dim_checker(named_x)
    with pytest.raises(TypeError):
        _named_m1bm1_dim_checker(named_x)


def test_named_float_tensor():
    @typeguard.typechecked
    def func(x: NamedFloatTensorType["a", "b":3]):
        pass

    x = torch.rand(2, 3, names=("a", "b"))
    y = torch.rand(2, 2, names=("a", "b"))
    z = torch.rand(2, 2, names=("a", "c"))
    w = torch.rand(2, 3)
    a = torch.rand(2, 2, names=("a", None))
    b = torch.rand(2, 3, names=("a", "b")).int()

    func(x)
    with pytest.raises(TypeError):
        func(y)
    with pytest.raises(TypeError):
        func(z)
    with pytest.raises(TypeError):
        func(w)
    with pytest.raises(TypeError):
        func(a)
    with pytest.raises(TypeError):
        func(b)
