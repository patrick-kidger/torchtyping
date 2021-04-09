import pytest
import torch
from torchtyping import TensorType
import typeguard


a = b = c = None


def test_fixed_int_dim():
    @typeguard.typechecked
    def _3_dim_checker(x: TensorType[3]):
        pass

    @typeguard.typechecked
    def _3m1_dim_checker(x: TensorType[3, -1]):
        pass

    @typeguard.typechecked
    def _4_dim_checker(x: TensorType[4]):
        pass

    @typeguard.typechecked
    def _4m1_dim_checker(x: TensorType[4, -1]):
        pass

    @typeguard.typechecked
    def _m14_dim_checker(x: TensorType[-1, 4]):
        pass

    @typeguard.typechecked
    def _m1m1_dim_checker(x: TensorType[-1, -1]):
        pass

    @typeguard.typechecked
    def _34_dim_checker(x: TensorType[3, 4]):
        pass

    @typeguard.typechecked
    def _34m1_dim_checker(x: TensorType[3, 4, -1]):
        pass

    @typeguard.typechecked
    def _m14m1_dim_checker(x: TensorType[-1, 4, -1]):
        pass

    x = torch.rand(3)
    _3_dim_checker(x)
    with pytest.raises(TypeError):
        _3m1_dim_checker(x)
    with pytest.raises(TypeError):
        _4_dim_checker(x)
    with pytest.raises(TypeError):
        _4m1_dim_checker(x)
    with pytest.raises(TypeError):
        _m14_dim_checker(x)
    with pytest.raises(TypeError):
        _m1m1_dim_checker(x)
    with pytest.raises(TypeError):
        _34_dim_checker(x)
    with pytest.raises(TypeError):
        _34m1_dim_checker(x)
    with pytest.raises(TypeError):
        _m14m1_dim_checker(x)

    x = torch.rand(3, 4)
    _3m1_dim_checker(x)
    _m14_dim_checker(x)
    _m1m1_dim_checker(x)
    _34_dim_checker(x)
    with pytest.raises(TypeError):
        _3_dim_checker(x)
    with pytest.raises(TypeError):
        _4_dim_checker(x)
    with pytest.raises(TypeError):
        _4m1_dim_checker(x)
    with pytest.raises(TypeError):
        _34m1_dim_checker(x)
    with pytest.raises(TypeError):
        _m14m1_dim_checker(x)

    x = torch.rand(4, 3)
    _4m1_dim_checker(x)
    _m1m1_dim_checker(x)
    with pytest.raises(TypeError):
        _3_dim_checker(x)
    with pytest.raises(TypeError):
        _3m1_dim_checker(x)
    with pytest.raises(TypeError):
        _4_dim_checker(x)
    with pytest.raises(TypeError):
        _m14_dim_checker(x)
    with pytest.raises(TypeError):
        _34_dim_checker(x)
    with pytest.raises(TypeError):
        _34m1_dim_checker(x)
    with pytest.raises(TypeError):
        _m14m1_dim_checker(x)


def test_str_dim():
    @typeguard.typechecked
    def _a_dim_checker(x: TensorType["a"]):
        pass

    @typeguard.typechecked
    def _ab_dim_checker(x: TensorType["a", "b"]):
        pass

    @typeguard.typechecked
    def _abc_dim_checker(x: TensorType["a", "b", "c"]):
        pass

    @typeguard.typechecked
    def _cb_dim_checker(x: TensorType["c", "b"]):
        pass

    @typeguard.typechecked
    def _am1_dim_checker(x: TensorType["a", -1]):
        pass

    @typeguard.typechecked
    def _m1b_dim_checker(x: TensorType[-1, "b"]):
        pass

    @typeguard.typechecked
    def _abm1_dim_checker(x: TensorType["a", "b", -1]):
        pass

    @typeguard.typechecked
    def _m1bm1_dim_checker(x: TensorType[-1, "b", -1]):
        pass

    x = torch.rand(3, 4)
    _ab_dim_checker(x)
    _cb_dim_checker(x)
    _am1_dim_checker(x)
    _m1b_dim_checker(x)
    with pytest.raises(TypeError):
        _a_dim_checker(x)
    with pytest.raises(TypeError):
        _abc_dim_checker(x)
    with pytest.raises(TypeError):
        _abm1_dim_checker(x)
    with pytest.raises(TypeError):
        _m1bm1_dim_checker(x)


def test_int_str_dim():
    @typeguard.typechecked
    def _a_dim_checker1(x: TensorType["a":3]):
        pass

    @typeguard.typechecked
    def _a_dim_checker2(x: TensorType["a":-1]):
        pass

    @typeguard.typechecked
    def _ab_dim_checker1(x: TensorType["a":3, "b":4]):
        pass

    @typeguard.typechecked
    def _ab_dim_checker2(x: TensorType["a":3, "b":-1]):
        pass

    @typeguard.typechecked
    def _ab_dim_checker3(x: TensorType["a":-1, "b":4]):
        pass

    @typeguard.typechecked
    def _ab_dim_checker4(x: TensorType["a":3, "b"]):
        pass

    @typeguard.typechecked
    def _ab_dim_checker5(x: TensorType["a", "b":4]):
        pass

    @typeguard.typechecked
    def _ab_dim_checker6(x: TensorType["a":5, "b":4]):
        pass

    @typeguard.typechecked
    def _ab_dim_checker7(x: TensorType["a":5, "b":-1]):
        pass

    @typeguard.typechecked
    def _m1b_dim_checker(x: TensorType[-1, "b":4]):
        pass

    @typeguard.typechecked
    def _abm1_dim_checker(x: TensorType["a":3, "b":4, -1]):
        pass

    @typeguard.typechecked
    def _m1bm1_dim_checker(x: TensorType[-1, "b":4, -1]):
        pass

    x = torch.rand(3, 4)
    _ab_dim_checker1(x)
    _ab_dim_checker2(x)
    _ab_dim_checker3(x)
    _ab_dim_checker4(x)
    _ab_dim_checker5(x)
    _m1b_dim_checker(x)
    with pytest.raises(TypeError):
        _a_dim_checker1(x)
    with pytest.raises(TypeError):
        _a_dim_checker2(x)
    with pytest.raises(TypeError):
        _ab_dim_checker6(x)
    with pytest.raises(TypeError):
        _ab_dim_checker7(x)
    with pytest.raises(TypeError):
        _abm1_dim_checker(x)
    with pytest.raises(TypeError):
        _m1bm1_dim_checker(x)


def test_return():
    @typeguard.typechecked
    def f1(x: TensorType["b":4]) -> TensorType["b":4]:
        return torch.rand(3)

    @typeguard.typechecked
    def f2(x: TensorType["b"]) -> TensorType["b":4]:
        return torch.rand(3)

    @typeguard.typechecked
    def f3(x: TensorType[4]) -> TensorType["b":4]:
        return torch.rand(3)

    @typeguard.typechecked
    def f4(x: TensorType["b":4]) -> TensorType["b"]:
        return torch.rand(3)

    @typeguard.typechecked
    def f5(x: TensorType["b"]) -> TensorType["b"]:
        return torch.rand(3)

    @typeguard.typechecked
    def f6(x: TensorType[4]) -> TensorType["b"]:
        return torch.rand(3)

    @typeguard.typechecked
    def f7(x: TensorType["b":4]) -> TensorType[4]:
        return torch.rand(3)

    @typeguard.typechecked
    def f8(x: TensorType["b"]) -> TensorType[4]:
        return torch.rand(3)

    @typeguard.typechecked
    def f9(x: TensorType[4]) -> TensorType[4]:
        return torch.rand(3)

    with pytest.raises(TypeError):
        f1(torch.rand(3))
    with pytest.raises(TypeError):
        f2(torch.rand(3))
    with pytest.raises(TypeError):
        f3(torch.rand(3))
    with pytest.raises(TypeError):
        f4(torch.rand(3))
    f5(torch.rand(3))
    with pytest.raises(TypeError):
        f6(torch.rand(3))
    with pytest.raises(TypeError):
        f7(torch.rand(3))
    with pytest.raises(TypeError):
        f8(torch.rand(3))
    with pytest.raises(TypeError):
        f9(torch.rand(3))

    with pytest.raises(TypeError):
        f1(torch.rand(4))
    with pytest.raises(TypeError):
        f2(torch.rand(4))
    with pytest.raises(TypeError):
        f3(torch.rand(4))
    with pytest.raises(TypeError):
        f4(torch.rand(4))
    with pytest.raises(TypeError):
        f5(torch.rand(4))
    f6(torch.rand(4))
    with pytest.raises(TypeError):
        f7(torch.rand(4))
    with pytest.raises(TypeError):
        f8(torch.rand(4))
    with pytest.raises(TypeError):
        f9(torch.rand(4))
