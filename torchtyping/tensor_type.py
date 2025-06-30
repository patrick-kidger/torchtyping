from __future__ import annotations

import sys
import torch

from .tensor_details import (
    _Dim,
    _no_name,
    is_named,
    DtypeDetail,
    LayoutDetail,
    ShapeDetail,
    TensorDetail,
)
from .utils import frozendict

from typing import Any, NoReturn, Generic, TypeVarTuple, Unpack

Ts = TypeVarTuple("Ts")

# Annotated is available in python version 3.9 (PEP 593)
if sys.version_info >= (3, 9):
    from typing import Annotated
else:
    # Else python version is lower than 3.9
    # we import Annotated from typing_annotations
    from typing_extensions import Annotated

# Not Type[Annotated...] as we want to use this in instance checks.
_AnnotatedType = type(Annotated[torch.Tensor, ...])


# For use when we have a plain TensorType, without any [].
class _TensorTypeMeta(type(torch.Tensor)):
    def __instancecheck__(cls, obj: Any) -> bool:
        return isinstance(obj, cls.base_cls)


# Inherit from torch.Tensor so that IDEs are happy to find methods on functions
# annotated as TensorTypes.
class TensorType(torch.Tensor, Generic[Unpack[Ts]], metaclass=_TensorTypeMeta):
    base_cls = torch.Tensor

    def __new__(cls, *args, **kwargs) -> NoReturn:
        raise RuntimeError(f"Class {cls.__name__} cannot be instantiated.")

    @staticmethod
    def _type_error(item: Any) -> NoReturn:
        raise TypeError(f"{item} not a valid type argument.")

    @classmethod
    def _convert_shape_element(cls, item_i: Any) -> _Dim:
        if isinstance(item_i, int) and not isinstance(item_i, bool):
            return _Dim(name=_no_name, size=item_i)
        elif isinstance(item_i, str):
            return _Dim(name=item_i, size=-1)
        elif item_i is None:
            return _Dim(name=None, size=-1)
        elif isinstance(item_i, slice):
            if item_i.step is not None:
                cls._type_error(item_i)
            if item_i.start is not None and not isinstance(item_i.start, str):
                cls._type_error(item_i)
            if item_i.stop is not ... and not isinstance(item_i.stop, (int, str)):
                cls._type_error(item_i)
            if item_i.start is None and item_i.stop is ...:
                cls._type_error(item_i)
            return _Dim(name=item_i.start, size=item_i.stop)
        elif item_i is ...:
            return _Dim(name=_no_name, size=...)
        elif item_i is Any:
            return _Dim(name=_no_name, size=-1)
        else:
            cls._type_error(item_i)

    @staticmethod
    def _convert_dtype_element(item_i: Any) -> torch.dtype:
        if item_i is int:
            return torch.long
        elif item_i is float:
            return torch.get_default_dtype()
        elif item_i is bool:
            return torch.bool
        else:
            return item_i

    def __class_getitem__(cls, item: Any) -> _AnnotatedType:
        if isinstance(item, tuple):
            if len(item) == 0:
                item = ((),)
        else:
            item = (item,)

        scalar_shape = False
        not_ellipsis = False
        not_named_ellipsis = False
        check_names = False
        dims = []
        dtypes = []
        layouts = []
        details = []
        for item_i in item:
            if isinstance(item_i, (int, str, slice)) or item_i in (None, ..., Any):
                item_i = cls._convert_shape_element(item_i)
                if item_i.size is ...:
                    # Supporting an arbitrary number of Ellipsis in arbitrary
                    # locations feels concerningly close to writing a regex
                    # parser and I definitely don't have time for that.
                    if not_ellipsis:
                        raise NotImplementedError(
                            "Having dimensions to the left of `...` is not currently "
                            "supported."
                        )
                    if item_i.name is None:
                        if not_named_ellipsis:
                            raise NotImplementedError(
                                "Having named `...` to the left of unnamed `...` is "
                                "not currently supported."
                            )
                    else:
                        not_named_ellipsis = True
                else:
                    not_ellipsis = True
                dims.append(item_i)
            elif isinstance(item_i, tuple):
                if len(item_i) == 0:
                    scalar_shape = True
                else:
                    cls._type_error(item_i)
            elif item_i in (int, bool, float) or isinstance(item_i, torch.dtype):
                dtypes.append(cls._convert_dtype_element(item_i))
            elif isinstance(item_i, torch.layout):
                layouts.append(item_i)
            elif item_i is is_named:
                check_names = True
            elif isinstance(item_i, TensorDetail):
                details.append(item_i)
            else:
                cls._type_error(item_i)

        if scalar_shape:
            if len(dims) != 0:
                cls._type_error(item)
        else:
            if len(dims) == 0:
                dims = None

        pre_details = []
        if dims is not None:
            pre_details.append(ShapeDetail(dims=dims, check_names=check_names))

        if len(dtypes) == 0:
            pass
        elif len(dtypes) == 1:
            pre_details.append(DtypeDetail(dtype=dtypes[0]))
        else:
            raise TypeError("Cannot have multiple dtypes.")

        if len(layouts) == 0:
            pass
        elif len(layouts) == 1:
            pre_details.append(LayoutDetail(layout=layouts[0]))
        else:
            raise TypeError("Cannot have multiple layouts.")

        details = tuple(pre_details + details)

        assert len(details) > 0

        # Frozen dict needed for Union[TensorType[...], ...], as Union hashes its
        # arguments.
        return Annotated[
            cls.base_cls,
            frozendict(
                {"__torchtyping__": True, "details": details, "cls_name": cls.__name__}
            ),
        ]
