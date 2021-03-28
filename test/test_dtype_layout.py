import pytest
import torch
from torchtyping import TensorType
import typeguard

from typing import Union


@typeguard.typechecked
def _float_checker(x: TensorType[float]):
    pass


@typeguard.typechecked
def _int_checker(x: TensorType[int]):
    pass


@typeguard.typechecked
def _union_int_float_checker(x: Union[TensorType[float], TensorType[int]]):
    pass


def test_float_dtype():
    x = torch.rand(2)
    _float_checker(x)
    _union_int_float_checker(x)
    with pytest.raises(TypeError):
        _int_checker(x)


def test_int_dtype():
    x = torch.tensor(2)
    _int_checker(x)
    _union_int_float_checker(x)
    with pytest.raises(TypeError):
        _float_checker(x)


@typeguard.typechecked
def _strided_checker(x: TensorType[torch.strided]):
    pass


@typeguard.typechecked
def _sparse_coo_checker(x: TensorType[torch.sparse_coo]):
    pass


def test_strided_layout():
    x = torch.rand(2)
    _strided_checker(x)
    with pytest.raises(TypeError):
        _sparse_coo_checker(x)


def test_sparse_coo_layout():
    x = torch.rand(2).to_sparse()
    _sparse_coo_checker(x)
    with pytest.raises(TypeError):
        _strided_checker(x)
