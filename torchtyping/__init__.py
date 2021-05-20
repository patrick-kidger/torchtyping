from .tensor_details import (
    DtypeDetail,
    is_float,
    is_named,
    LayoutDetail,
    ShapeDetail,
    TensorDetail,
)

from .tensor_type import TensorType
from .typechecker import patch_typeguard

__version__ = "0.1.2"
