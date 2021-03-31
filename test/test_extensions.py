import pytest
from torch import rand, Tensor
from torchtyping import TensorDetail, TensorType
from typeguard import typechecked

good = foo = None

# Write the extension


class FooDetail(TensorDetail):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def check(self, tensor: Tensor) -> bool:
        return hasattr(tensor, "foo") and tensor.foo == self.foo

    # reprs used in error messages when the check is failed

    def __repr__(self) -> str:
        return f"FooDetail({self.value})"

    @classmethod
    def tensor_repr(cls, tensor: Tensor) -> str:
        # Should return a representation of the tensor with respect
        # to what this detail is checking
        if hasattr(tensor, "foo"):
            return f"FooDetail({tensor.foo})"
        else:
            return ""


# Test the extension


@typechecked
def foo_checker(tensor: TensorType[float, FooDetail("good-foo")]):
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
    x = rand(2).int()
    x.foo = "good-foo"
    foo_checker(x)


def test_extensions():
    valid_foo()
    with pytest.raises(TypeError):
        invalid_foo_one()
    with pytest.raises(TypeError):
        invalid_foo_two()
