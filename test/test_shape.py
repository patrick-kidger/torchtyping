import pytest
from typing import Any
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


def test_any_dim():
    @typeguard.typechecked
    def _3any_dim_checker(x: TensorType[3, Any]):
        pass

    @typeguard.typechecked
    def _any4_dim_checker(x: TensorType[Any, 4]):
        pass

    @typeguard.typechecked
    def _anyany_dim_checker(x: TensorType[Any, Any]):
        pass

    @typeguard.typechecked
    def _34any_dim_checker(x: TensorType[3, 4, Any]):
        pass

    @typeguard.typechecked
    def _any4any_dim_checker(x: TensorType[Any, 4, Any]):
        pass

    x = torch.rand(3)
    with pytest.raises(TypeError):
        _3any_dim_checker(x)
    with pytest.raises(TypeError):
        _any4_dim_checker(x)
    with pytest.raises(TypeError):
        _anyany_dim_checker(x)
    with pytest.raises(TypeError):
        _34any_dim_checker(x)
    with pytest.raises(TypeError):
        _any4any_dim_checker(x)

    x = torch.rand((3, 4))
    _3any_dim_checker(x)
    _any4_dim_checker(x)
    _anyany_dim_checker(x)

    x = torch.rand((4, 5))
    with pytest.raises(TypeError):
        _any4_dim_checker(x)

    x = torch.rand(4, 5)
    with pytest.raises(TypeError):
        _3any_dim_checker(x)

    x = torch.rand((3, 4, 5))
    _34any_dim_checker(x)
    _any4any_dim_checker(x)

    x = torch.rand((3, 5, 5))
    with pytest.raises(TypeError):
        x = _any4any_dim_checker(x)
    with pytest.raises(TypeError):
        _34any_dim_checker(x)