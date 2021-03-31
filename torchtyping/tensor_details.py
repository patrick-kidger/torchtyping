from __future__ import annotations

import abc
import collections
import torch

from typing import Optional, Union


ellipsis = type(...)


class TensorDetail(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def check(self, tensor: torch.Tensor) -> bool:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def tensor_repr(cls, tensor: torch.Tensor) -> str:
        raise NotImplementedError


# inheriting from typing.NamedTuple crashes typeguard
class _Dim(collections.namedtuple("_Dim", ["name", "size"])):
    name: Union[None, str]
    size: Union[ellipsis, int]

    def __repr__(self) -> str:
        if self.name is None:
            if self.size is ...:
                return "..."
            else:
                return repr(self.size)
        else:
            if self.size is ...:
                return f"{self.name}: ..."
            elif self.size == -1:
                return repr(self.name)
            else:
                return f"{self.name}: {self.size}"


class ShapeDetail(TensorDetail):
    def __init__(self, *, dims: list[_Dim], check_names: bool, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dims = dims
        self.check_names = check_names

    def __repr__(self) -> str:
        return repr(tuple(self.dims))

    def check(self, tensor: torch.Tensor) -> bool:
        self_names = [self_dim.name for self_dim in self.dims]
        self_shape = [self_dim.size for self_dim in self.dims]

        if ... in self_shape:
            if sum(1 for size in self_shape if size is not ...) > len(tensor.names):
                return False
        else:
            if len(self_shape) != len(tensor.names):
                return False

        for self_name, self_size, tensor_name, tensor_size in zip(
            reversed(self_names),
            reversed(self_shape),
            reversed(tensor.names),
            reversed(tensor.shape),
        ):
            if self_size is ...:
                # This assumes that Ellipses only occur on the left hand edge.
                # So once we hit one we're done.
                break

            if self.check_names and self_name is not None and self_name != tensor_name:
                return False
            if self_size not in (-1, tensor_size):
                return False

        return True

    @classmethod
    def tensor_repr(cls, tensor: torch.Tensor) -> str:
        dims = []
        for name, size in zip(tensor.names, tensor.shape):
            dims.append(_Dim(name=name, size=size))
        return repr(cls(dims=dims))

    def update(
        self,
        *,
        dims: Optional[list[_Dim]] = None,
        check_names: Optional[bool] = None,
        **kwargs,
    ) -> ShapeDetail:
        dims = self.dims if dims is None else dims
        check_names = self.check_names if check_names is None else check_names
        return type(self)(dims=dims, check_names=check_names, **kwargs)


class DtypeDetail(TensorDetail):
    def __init__(self, *, dtype, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dtype = dtype

    def __repr__(self) -> str:
        return repr(self.dtype)

    def check(self, tensor: torch.Tensor) -> bool:
        return self.dtype == tensor.dtype

    @classmethod
    def tensor_repr(cls, tensor: torch.Tensor) -> str:
        return repr(cls(tensor.dtype))


class LayoutDetail(TensorDetail):
    def __init__(self, *, layout, **kwargs) -> None:
        super().__init__(**kwargs)
        self.layout = layout

    def __repr__(self) -> str:
        return repr(self.layout)

    def check(self, tensor: torch.Tensor) -> bool:
        return self.layout == tensor.layout

    @classmethod
    def tensor_repr(cls, tensor: torch.Tensor) -> str:
        return repr(cls(tensor.layout))


class _FloatDetail(TensorDetail):
    def __repr__(self) -> str:
        return "float_detail"

    def check(self, tensor: torch.Tensor) -> bool:
        return tensor.is_floating_point()

    @classmethod
    def tensor_repr(cls, tensor: torch.Tensor) -> str:
        return "float_detail" if tensor.is_floating_point() else ""


class _NamedTensorDetail(TensorDetail):
    def __repr__(self) -> str:
        return "named_detail"

    # named_detail is special-cased and consumed by TensorType.
    # It's a bit of an odd exception.
    def check(self, tensor: torch.Tensor) -> bool:
        raise RuntimeError

    @classmethod
    def tensor_repr(cls, tensor: torch.Tensor) -> str:
        raise RuntimeError


float_detail = _FloatDetail()  # singleton flag
named_detail = _NamedTensorDetail()  # singleton flag
