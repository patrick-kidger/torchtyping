from __future__ import annotations

import collections as co
import threading
import torch

from typing import Any, Dict, List, Tuple, Union


NoneType = type(None)
ellipsis = type(...)

        
class _TensorTypeMeta(type):
    _cache = {}
    
    def __new__(mcs, name, bases, dict):
        for base in bases:
            if base._torchtyping_is_getitem_subclass:
                raise TypeError("Cannot subclass TensorType[...anything here...].")
        return super().__new__(name, bases, dict)
        
    def __repr__(cls) -> str:
        return cls.__name__
    
    def __instancecheck__(cls, instance: Any) -> bool:
        return cls.check(instance)
        
    def __getitem__(cls, item: Any) -> _TensorTypeMeta:
        if cls._torchtyping_is_getitem_subclass:
            assert len(cls.__bases__) == 1
            base_cls = cls.__bases__[0]
        else:
            base_cls = cls
        name = base_cls.__name__
        dict = cls.getitem(item)
        intersection = cls._torchtyping_fields.intersection(dict.keys())
        if intersection:
            raise TypeError(f"Overwriting {intersection} fields.")
        fields = cls._torchtyping_fields | dict.keys()
        key = [base_cls]
        for field in sorted(fields):
            value = dict[field]
            if value is not None:
                name += f"[{field}={value}]"
            key.append((field, value))
        key = tuple(key)
        try:
            return type(cls)._cache[key]
        except KeyError:
            dict["_torchtyping_is_getitem_subclass"] = True
            dict["_torchtyping_fields"] = fields
            out = type(cls)(name, (base_cls,), dict)
            type(cls)._cache[key] = out
            return out
            
            
_dimension_resolution = theading.local()
_dimension_resolution.details: Optional[Dict[str, int]] = None

        
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
            
        details = _dimension_resolution.details
        if details is None:
            # This may feature some dimensions of size -1, or some ...
            # indicating multiple dimensions.
            cls_names = [cls_dim.start for cls_dim in cls.dims]
            cls_shape = [cls_dim.stop for cls_dim in cls.dims]
        else:
            # However if we have some details available then we can try
            # to fill in some of the unspecified dimension sizes (-1) and
            # number of dimensions (...).
            reverse_cls_names = []
            reverse_cls_shape = []
            for dim in reversed(cls.dims):
                name = dim.start
                size = dim.stop
                num_dims = 1
                if name is not None:
                    if size == -1:
                        size = details[name]
                    if size is ...:
                        # This assumes that named Ellipses only occur to the right of
                        # unnamed Ellipses, to avoid filling in Ellipses that occur to
                        # the left of other Ellipses.
                        size = -1
                        num_dims = details[name]
                for _ in range(num_dims):
                    reverse_cls_names.append(name)
                    reverse_cls_shape.append(size)
            cls_names = reversed(reverse_cls_names)
            cls_shape = reversed(reverse_cls_shape)
                
        for cls_name, cls_size, instance_name, instance_size in zip(reversed(cls_names), reversed(cls_shape), reversed(instance.names), reversed(instance.shape)):
            if cls_size is ...:
                # This assumes that Ellipses only occur on the left hand edge.
                # So once we hit one we're done.
                break

            if cls._torchtyping_validate_named_tensor and cls_name is not None and instance_name is not None and cls_name != instance_name:
                return False
            if cls_size not in (-1, instance_dim):
                return False
                
        return True
        
    # public:
    
    dims: Optional[Tuple[slice, ...]] = None  # slice[Union[None, str], Union[ellipsis, int], None]
    dtype: Optional[torch.dtype] = None
    layout: Optional[torch.layout] = None
        
    @classmethod
    def check(cls, instance: Any) -> bool:
        return isinstance(instance, torch.Tensor) and (cls.dtype in (None, instance.dtype)) and (cls.layout in (None, instance.layout)) and cls._check_dims(instance)

    @classmethod
    def getitem(cls, item: Any) -> Dict[str, Any]:
    
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
        #     TensorType[..., "a": 3, "b"]  # names (..., "a", "b") and shape (..., 3, Any)
        #   to name the arbitrary number of leftmost dimensions:
        #     TensorType["name": ..., "a": 3, "b": 4]  # names (..., "a", "b") and shape (..., 3, 4).
        #                                              # The `...` group is checked against the other
        #                                              # function arguments with the same "name".
        #     TensorType[..., "name": ..., 3, 4]       # No name checking. Shape (..., 3, 4).
        #                                              # The first `...` group can be of any size. The
        #                                              # second one is checked against other function
        #                                              # arguments with the same "name".
        #
        #   Note that names are validated if and only if using NamedTensorType.
        #     TensorType does not perform this checking. Naming dimensions are
        #     still useful for checking sizes against other arguments.
        #
        #   Note that ellipses are currently only supported in the leftmost positions.
        #########
        
        if isinstance(item, int):
            item = (slice(None, item),)
        elif isinstance(item, str):
            item = (slice(item, -1),)
        elif item is ...:
            item = (slice(None, ...),)
            
        if isinstance(item, tuple) and all(isinstance(item_i, slice) and isinstance(item_i.start, (NoneType, str)) and isinstance(item_i.stop, (ellipsis, int)) and item_i.step is None for item_i in item):
            not_ellipsis = False
            not_named_ellipsis = False
            for item_i in item:
                name = item_i.start
                size = item_i.stop
                if size is ...:
                    # Supporting an arbitrary number of Ellipsis in arbitrary
                    # locations feels concerningly close to writing a regex
                    # parser and I definitely don't have time for that.
                    if not_ellipsis:
                        raise NotImplementedError("Having dimensions to the left of `...` is not currently supported.")
                    if name is None:
                        if not_named_ellipsis:
                            raise NotImplementedError("Having named `...` to the left of unnamed `...` is not currently supported.")
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
            raise TypeError(f"{item} not a valid type argument.")
