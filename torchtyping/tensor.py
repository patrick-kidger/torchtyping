from __future__ import annotations

import torch

from typing import Any, Tuple
        
        
class _TensorTypeMeta(type):
    _cache = {}
        
    def __repr__(cls) -> str:
        return cls.__name__
    
    def __instancecheck__(cls, instance: Any) -> bool:
        return cls.check(instance)
        
    def __getitem__(cls, item: Any) -> _TensorTypeMeta:
        if item is None:
            # Corresponding to how None is allow in TensorType.getitem: it has a
            # special value there, so we disallow it here.
            raise ValueError(f"{item} not a valid type argument.")

        if cls._is_getitem_subclass:
            assert len(cls.__bases__) == 1
            base_cls = cls.__bases__[0]
        else:
            base_cls = cls
        name = base_cls.__name__
        dict = cls.getitem(item)
        for field in cls.fields():
            value = dict[field]
            if value is not None:
                name += f"[{field}={value}]"
        dict["_is_getitem_subclass"] = True
        try:
            return type(cls)._cache[name, base_cls]
        except KeyError:
            out = type(cls)(name, (base_cls,), dict)
            type(cls)._cache[name, base_cls] = out
            return out
        
        
class TensorType(metaclass=_TensorTypeMeta):
    _is_getitem_subclass = False
    
    def __new__(cls, *args, **kwargs):
        raise RuntimeError(f"Class {cls.__name__} cannot be instantiated.")
    
    dtype = None
    layout = None
        
    @classmethod
    def fields(cls) -> Tuple[str]:
        return ('dtype', 'layout')
        
    @classmethod
    def check(cls, instance: Any) -> bool:
        return isinstance(instance, torch.Tensor) and (cls.dtype in (None, instance.dtype)) and (cls.layout in (None, instance.layout))
        
    @classmethod
    def getitem(cls, item: Any) -> TensorType:
        dtype = cls.dtype
        layout = cls.layout
        
        if item is int:
            dtype = torch.long
        elif item is float:
            dtype = torch.get_default_dtype()
        elif item is bool:
            dtype = torch.bool
        elif isinstance(item, torch.dtype):
            dtype = item
        elif isinstance(item, torch.layout):
            layout = item
        elif item is None:
            pass  # To allow subclasses to pass item=None to indicate no further processing.
        else:
            raise ValueError(f"{item} not a valid type argument.")
            
        return dict(dtype=dtype, layout=layout)
