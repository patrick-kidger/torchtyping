import pytest
import torch
from torchtyping import NamedTensorType
import typeguard


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


def test_named_str_dim():
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
