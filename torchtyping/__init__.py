# End users will basically never need to touch any of these.
# Maybe TensorDetail if they want to create their own details.
# Still, they're provided here -- public but not documented --
# for anyone who wants to inherit from one of them.
from .tensor_details import (
    float_detail,
    DtypeDetail,
    LayoutDetail,
    named_detail,
    ShapeDetail,
    TensorDetail,
)

from .tensor_type import TensorType
from .typechecker import patch_typeguard

__version__ = "0.1.0"
