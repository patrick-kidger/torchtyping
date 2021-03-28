import torch

from .tensor_type import TensorType

from typing import Any


class FloatTensorType(TensorType):
    @classmethod
    def check(cls, instance: Any) -> bool:
        check = super().check(instance)
        if not instance.is_floating_point():
            return False
        return check


class NamedTensorType(TensorType):
    _torchtyping_validate_named_tensor = True


class NamedFloatTensorType(FloatTensorType, NamedTensorType):
    pass
