#####################
# torchtyping is designed to be highly extensible.
#
# Extensions are performed by subclassing TensorType, and overriding its `check` and
# `getitem` methods.
#
# `check` performs the instance check, if this is an instance is of this class. It
# should return a bool for whether the check passes.
#
# `getitem` specifies how the [] notation should update the attributes of the class. It
# should return a dictionary, whose keys correspond to the attributes to update, and
# whose values correspond to the values those attributes should take. A subclass will be
# created with those attributes.
#
# Here's an example extending TensorType to check that the passed tensor has an extra
# attribute `foo` whose value must be the string "good-foo".
#####################

from torch import rand, tensor
from torchtyping import TensorType
from typeguard import typechecked

from typing import Any, Dict, Optional


class FooType:
    def __init__(self, value):
        self.value = value


class FooTensorType(TensorType):
    foo: Optional[str] = None
    
    @classmethod
    def check(cls, instance: Any) -> bool:
        check = super().check(instance)
        if cls.foo is not None:
            check = check and hasattr(instance, "foo") and instance.foo == cls.foo
        return check
        
    @classmethod
    def getitem(cls, item: Any) -> Dict[str, Any]
        if isinstance(item, FooType):
            return {"foo": item.value}
        else:
            return super().getitem(item)


@typechecked
def foo_checker(tensor: FooTensorType[float][FooType("good-foo")]):
    pass

    
def valid_foo():
    x = rand(3)
    x.foo = "good-foo"
    foo_checker(x)


def invalid_foo_one():
    x = rand(3)
    x.foo = "bad-foo"
    foo_checker(x)
    
    
def invalid_foo_two():
    x = tensor([1, 2])  # integer type
    x.foo = "good-foo"
    foo_checker(x)
