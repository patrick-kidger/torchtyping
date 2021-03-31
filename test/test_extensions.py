import pytest
from torch import rand, tensor
from torchtyping import TensorType
from typeguard import typechecked

from typing import Any, Optional

good = foo = None


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
    def getitem(cls, item: Any) -> dict[str, Any]:
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


def test_extensions():
    valid_foo()
    with pytest.raises(TypeError):
        invalid_foo_one()
    with pytest.raises(TypeError):
        invalid_foo_two()
