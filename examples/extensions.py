#####################
# torchtyping is designed to be highly extensible.
#####################

from __future__ import annotations

import torch
from torchtyping import TensorType
import typeguard

from typing import Any, Tuple


#####################
# It's possible to check any other property of a tensor, as well as just the defaults.
#
# Here we check that the tensor has an attribute called "foo" on it, which should
# take a particular value.
#####################
class TensorTypeFooChecker(TensorType):
    foo = None
    
    @classmethod
    def fields(cls) -> Tuple[str]:
        return super().fields() + ('foo',)
    
    @classmethod
    def check(cls, instance: Any) -> bool:
        check = super().check(instance)
        if cls.foo is not None:
            check = check and hasattr(instance, "foo") and instance.foo == cls.foo
        return check
        
    @classmethod
    def getitem(cls, item: Any) -> TensorTypeFooChecker:
        foo = cls.foo
        if isinstance(item, slice):
            if item.start == "foo":
                foo = item.stop
                item = None
        dict = super().getitem(item)
        dict.update(foo=foo)
        return dict


@typeguard.typechecked
def foo_checker(tensor: TensorTypeFooChecker["foo":"good-foo"][float]):
    pass
    
    
def valid_foo():
    x = torch.rand(3)
    x.foo = "good-foo"
    foo_checker(x)
    
    
def invalid_foo():
    x = torch.rand(3)
    x.foo = "bad-foo"
    foo_checker(x)
