from __future__ import annotations

import torch

from typing import Any, NamedTuple, NoReturn, Optional, Union


ellipsis = type(...)


class _TensorTypeMeta(type):
    _cache = {}

    def __new__(mcs, name, bases, dict):
        for base in bases:
            if base._torchtyping_is_getitem_subclass:
                raise TypeError("Cannot subclass TensorType[...anything here...].")
        return super().__new__(mcs, name, bases, dict)

    def __repr__(cls) -> str:
        return cls.__name__

    def __instancecheck__(cls, instance: Any) -> bool:
        return cls.check(instance)

    def __getitem__(cls, item: Any) -> _TensorTypeMeta:
        return cls._meta_getitem(item, disallow_overwrite=True)

    def update(cls, item: Any) -> _TensorTypeMeta:
        return cls._meta_getitem(item, disallow_overwrite=False)

    def _meta_getitem(cls, item: Any, disallow_overwrite: bool) -> _TensorTypeMeta:
        if cls._torchtyping_is_getitem_subclass:
            assert len(cls.__bases__) == 1
            base_cls = cls.__bases__[0]
        else:
            base_cls = cls
        name = base_cls.__name__
        dict = cls.getitem(item)

        if disallow_overwrite:
            intersection = cls._torchtyping_fields.intersection(dict.keys())
            if len(intersection) > 0:
                raise TypeError(f"Overwriting fields: {intersection}.")

        dict.update({field: getattr(cls, field) for field in cls._torchtyping_fields})
        key = [base_cls]
        for field in sorted(dict.keys()):
            value = dict[field]
            if value is not None:
                if isinstance(value, tuple):
                    if len(value) == 1:
                        name += f"[{value[0]}]"
                    else:
                        name += f"[{str(value)[1:-1]}]"
                else:
                    name += f"[{value}]"
            key.append((field, value))
        key = tuple(key)
        try:
            return type(cls)._cache[key]
        except KeyError:
            dict["_torchtyping_fields"] = set(dict.keys())
            dict["_torchtyping_is_getitem_subclass"] = True
            out = type(cls)(name, (base_cls,), dict)
            type(cls)._cache[key] = out
            return out


class _Dim(NamedTuple):
    name: Union[None, str]
    size: Union[ellipsis, int]

    def __repr__(self):
        if self.name is None:
            if self.size is ...:
                return "..."
            else:
                return str(self.size)
        else:
            if self.size is ...:
                return f"{self.name}: ..."
            elif self.size == -1:
                return self.name
            else:
                return f"{self.name}: {self.size}"


class TensorType(metaclass=_TensorTypeMeta):
    # private:

    _torchtyping_is_getitem_subclass: bool = False
    _torchtyping_fields = set()
    _torchtyping_validate_named_tensor = False

    def __new__(cls, *args, **kwargs):
        raise RuntimeError(f"Class {cls.__name__} cannot be instantiated.")

    @classmethod
    def _check_dims(cls, instance: torch.Tensor) -> bool:
        if cls.dims is None:
            return True

        cls_names = [cls_dim.name for cls_dim in cls.dims]
        cls_shape = [cls_dim.size for cls_dim in cls.dims]

        if len(cls_names) != len(instance.names):
            return False

        for cls_name, cls_size, instance_name, instance_size in zip(
            reversed(cls_names),
            reversed(cls_shape),
            reversed(instance.names),
            reversed(instance.shape),
        ):
            if cls_size is ...:
                # This assumes that Ellipses only occur on the left hand edge.
                # So once we hit one we're done.
                break

            if (
                cls._torchtyping_validate_named_tensor
                and cls_name is not None
                and cls_name != instance_name
            ):
                return False
            if cls_size not in (-1, instance_size):
                return False

        return True

    @staticmethod
    def _type_error(item: Any) -> NoReturn:
        raise TypeError(f"{item} not a valid type argument.")

    @classmethod
    def _convert_tuple_element(cls, item: Any) -> _Dim:
        if isinstance(item, int) and not isinstance(item, bool):
            return _Dim(name=None, size=item)
        elif isinstance(item, str):
            return _Dim(name=item, size=-1)
        elif isinstance(item, slice):
            if item.step is not None:
                cls._type_error(item)
            if item.start is None and item.stop is None:
                cls._type_error(item)
            return _Dim(name=item.start, size=item.stop)
        elif item is ...:
            return _Dim(name=None, size=...)
        elif isinstance(item, _Dim):
            return item
        else:
            cls._type_error(item)

    # public:

    dims: Optional[tuple[_Dim, ...]] = None
    dtype: Optional[torch.dtype] = None
    layout: Optional[torch.layout] = None

    @classmethod
    def check(cls, instance: Any) -> bool:
        return (
            isinstance(instance, torch.Tensor)
            and (cls.dtype in (None, instance.dtype))
            and (cls.layout in (None, instance.layout))
            and cls._check_dims(instance)
        )

    @classmethod
    def getitem(cls, item: Any) -> dict[str, Any]:

        #########
        # Dim:
        #
        # syntax is of the forms:
        #   to specify shape:
        #     TensorType[3]          # shape (3,)
        #     TensorType[3, 4]       # shape (3, 4)
        #     TensorType[-1, 5]      # shape (Any, 5)
        #     TensorType[-1, -1]     # shape (Any, Any)
        #   to specify names:
        #     TensorType["a"]        # names ("a",)
        #     TensorType["a", "b"]   # names ("a", "b")
        #     TensorType["a", -1]    # names ("a", Any)
        #   to specify both names and shapes:
        #     TensorType["a": 1]       # names ("a",) and shape (1,)
        #     TensorType["a", "b": 3]  # names ("a", "b") and shape (Any, 3)
        #     TensorType["a", -1, 3]   # names ("a", Any, Any) and shape (Any, Any, 3)
        #   to specify an arbitrary number of dimensions at the start:
        #     TensorType[..., 3, 4]         # shape (..., 3, 4)
        #     TensorType[..., "a": 3, "b"]  # names (..., "a", "b") and shape (..., 3, Any)             # noqa
        #   to name the arbitrary number of leftmost dimensions:
        #     TensorType["name": ..., "a": 3, "b": 4]  # names (..., "a", "b") and shape (..., 3, 4).   # noqa
        #                                              # The `...` group is checked against the other   # noqa
        #                                              # function arguments with the same "name".       # noqa
        #     TensorType[..., "name": ..., 3, 4]       # No name checking. Shape (..., 3, 4).           # noqa
        #                                              # The first `...` group can be of any size. The  # noqa
        #                                              # second one is checked against other function   # noqa
        #                                              # arguments with the same "name".                # noqa
        #
        #   Note that names are validated if and only if using NamedTensorType.
        #     TensorType does not perform this checking. Naming dimensions are
        #     still useful for checking sizes against other arguments.
        #
        #   Note that ellipses are currently only supported in the leftmost positions.
        #########

        if isinstance(item, (int, str, slice)) or item is ...:
            return {"dims": (cls._convert_tuple_element(item),)}
        elif isinstance(item, tuple):
            item = tuple(cls._convert_tuple_element(item_i) for item_i in item)
            not_ellipsis = False
            not_named_ellipsis = False
            for item_i in item:
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
            return {"dims": item}

        #########
        # Dtype:
        # syntax is of the form TensorType[float] or TensorType[torch.bool]
        #########

        elif item is int:
            return {"dtype": torch.long}
        elif item is float:
            return {"dtype": torch.get_default_dtype()}
        elif item is bool:
            return {"dtype": torch.bool}
        elif isinstance(item, torch.dtype):
            return {"dtype": item}

        #########
        # Layout:
        # syntax is of the form TensorType[torch.strided]
        #########

        elif isinstance(item, torch.layout):
            return {"layout": item}
        else:
            cls._type_error(item)
