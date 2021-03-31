import pytest
import torch
from torchtyping import TensorType, float_detail, named_detail
import typeguard


a = b = c = None


def test_float_tensor():
    @typeguard.typechecked
    def func1(x: TensorType[float_detail]):
        pass

    @typeguard.typechecked
    def func2(x: TensorType[2, 2, float_detail]):
        pass

    @typeguard.typechecked
    def func3(x: TensorType[float, float_detail]):
        pass

    @typeguard.typechecked
    def func4(x: TensorType[bool, float_detail]):
        pass

    @typeguard.typechecked
    def func5(x: TensorType["a":2, 2, float, float_detail]):
        pass

    @typeguard.typechecked
    def func6(x: TensorType[2, "b":2, torch.sparse_coo, float_detail]):
        pass

    x = torch.rand(2, 2)
    y = torch.rand(1)
    z = torch.tensor([[0, 1], [2, 3]])
    w = torch.rand(4).to_sparse()
    w1 = torch.rand(2, 2).to_sparse()
    w2 = torch.tensor([[0, 1], [2, 3]]).to_sparse()

    func1(x)
    func1(y)
    with pytest.raises(TypeError):
        func1(z)
    func1(w)
    func1(w1)
    with pytest.raises(TypeError):
        func1(w2)

    func2(x)
    with pytest.raises(TypeError):
        func2(y)
    with pytest.raises(TypeError):
        func2(z)
    with pytest.raises(TypeError):
        func2(w)
    func2(w1)
    with pytest.raises(TypeError):
        func2(w2)

    func3(x)
    func3(y)
    with pytest.raises(TypeError):
        func3(z)
    func3(w)
    func3(w1)
    with pytest.raises(TypeError):
        func3(w2)

    with pytest.raises(TypeError):
        func4(x)
    with pytest.raises(TypeError):
        func4(y)
    with pytest.raises(TypeError):
        func4(z)
    with pytest.raises(TypeError):
        func4(w)
    with pytest.raises(TypeError):
        func4(w1)
    with pytest.raises(TypeError):
        func4(w2)

    func5(x)
    with pytest.raises(TypeError):
        func5(y)
    with pytest.raises(TypeError):
        func5(z)
    with pytest.raises(TypeError):
        func5(w)
    func5(w1)
    with pytest.raises(TypeError):
        func5(w2)

    with pytest.raises(TypeError):
        func6(x)
    with pytest.raises(TypeError):
        func6(y)
    with pytest.raises(TypeError):
        func6(z)
    with pytest.raises(TypeError):
        func6(w)
    func6(w1)
    with pytest.raises(TypeError):
        func6(w2)


def test_named_tensor():
    @typeguard.typechecked
    def _named_a_dim_checker(x: TensorType["a", named_detail]):
        pass

    @typeguard.typechecked
    def _named_ab_dim_checker(x: TensorType["a", "b", named_detail]):
        pass

    @typeguard.typechecked
    def _named_abc_dim_checker(x: TensorType["a", "b", "c", named_detail]):
        pass

    @typeguard.typechecked
    def _named_cb_dim_checker(x: TensorType["c", "b", named_detail]):
        pass

    @typeguard.typechecked
    def _named_am1_dim_checker(x: TensorType["a", -1, named_detail]):
        pass

    @typeguard.typechecked
    def _named_m1b_dim_checker(x: TensorType[-1, "b", named_detail]):
        pass

    @typeguard.typechecked
    def _named_abm1_dim_checker(x: TensorType["a", "b", -1, named_detail]):
        pass

    @typeguard.typechecked
    def _named_m1bm1_dim_checker(x: TensorType[-1, "b", -1, named_detail]):
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
    def func(x: TensorType["a", "b":3, float_detail, named_detail]):
        pass

    x = torch.rand(2, 3, names=("a", "b"))
    y = torch.rand(2, 2, names=("a", "b"))
    z = torch.rand(2, 2, names=("a", "c"))
    w = torch.rand(2, 3)
    w1 = torch.rand(2, 2, names=("a", None))
    w2 = torch.rand(2, 3, names=("a", "b")).int()

    func(x)
    with pytest.raises(TypeError):
        func(y)
    with pytest.raises(TypeError):
        func(z)
    with pytest.raises(TypeError):
        func(w)
    with pytest.raises(TypeError):
        func(w1)
    with pytest.raises(TypeError):
        func(w2)
