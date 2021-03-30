# Further documentation

## Further API

`torchtyping` has a few other things in its API beyond that listed in the main [README](./README.md) file.

```python
torchtyping.NamedTensorType
```

By default the names associated with each dimension in `TensorType` are not checked against the dimension names used as part of being a [named tensor](https://pytorch.org/docs/stable/named_tensor.html). This is because named tensors aren't used very frequently, and often we still want to name the `TensorType` dimensions to check that they're the same size as other arguments.

`NamedTensorType` performs these additional checks. For example `func(x: NamedTensorType["batch", "channels"])` may be called with `torch.rand(4, 5, names=("batch", "channels"))`, but not `torch.rand(4, 5)`.

```python
torchtyping.FloatTensorType
torchtyping.NamedFloatTensorType
```

There's quite a few floating point types: `torch.float16`, `torch.bfloat16`, `torch.float32`, `torch.float64`, `torch.complex64`, `torch.complex128`. Frequently we're not that fussed which one we get.

These are a convenience to allow any such dtype.

## FAQ

**The runtime checking isn't working!**

Make sure that you've enable typeguard, either by decorating with `typeguard.typechecked`, or by using `typeguard.importhook.install_import_hook()`, or by using the pytest command line flags listed in the main [README](./README.md).

If you have done all of that, then feel free to raise an issue.

**Silencing spurious flake8 warnings.**

Running flake8 will produce spurious warnings for annotations using strings: `TensorType["batch"]` gives `F821 undefined name 'batch'`.

You can silence these en-masse just by creating a dummy `batch = None` anywhere in the file. (Or more laboriously, placing `# noqa: F821` on the relevant lines.)

**Does this work with mypy?**  

Mostly.

The functionality provided by `torchtyping` is [explicitly beyond the current scope of mypy](https://www.python.org/dev/peps/pep-0586/#true-dependent-types-integer-generics), so there's not much hope of tight integration.

But at the very least, `torchtyping` generally shouldn't break mypy. Put a `# type: ignore` comment on the same line in which you import `torchtyping` and you'll be good to go.

There is one exception: using `TensorType["string": value]` hits a bug in mypy and causes a crash. See the corresponding [mypy issue](https://github.com/python/mypy/issues/10266).

**What does `patch_typeguard()` actually do?**

This enables the extra consistency checks, that all named dimensions are the same across all tensors.

Without it, the checks will only be done at the level of individual tensors. e.g. 

```python
def func(x: TensorType["batch"], y: TensorType["batch"]):
    pass

func(torch.rand(5), torch.rand(7))
```

won't raise an error.  `rand(5)` matches `TensorType["batch"]` when considered in isolation, and `rand(7)` likewise matches `TensorType["batch"]` when considered in isolation.

**How to indicate a scalar Tensor?**

`TensorType[()]`

**Are nested annotations of the form `Blahblah[Moreblah[TensorType[...]]]` supported?**

Yes.

**Are multiple `...` supported?**

Yes. For example:

```python
def func(x:  TensorType["dim1": ..., "dim2": ...],
         y:  TensorType["dim2": ...]
        ) -> TensorType["dim1": ...]:
    sum_dims = [x - 1 for x in range(y.dim())]
    return (x * y).sum(dim=sum_dims)
```

**Trying to use `...` is raising a `NotImplementedError`**

Using `...` in a shape specification is currently only supported in the left-most places of a tensor shape. Supporting using `...` in other locations would be a fair bit more complicated.

**`TensorType[float]` corresponds to`float32` but `torch.rand(2).to(float)` produces `float64`**.

This is a deliberate asymmetry. `TensorType[float]` corresponds to `torch.get_default_dtype()`, as a convenience, but `.to(float)` always corresponds to `float64`. 

**Why is `TensorType` implemented the way it is / shouldn't it be implemented in a different way / why use `typeguard` rather than a different package / etc. ?**

The short answer is that this works around the various limitations elsewhere in the ecosystem.

Python does provide a comprehensive typing system that we could have used instead -- `typing.Annotated[torch.Tensor, ...]`,  would probably make the most sense -- but then there's no easy way to add the extra shape checking that gives `torchtyping` its _raison d'etre_.

## Custom extensions

Writing custom extensions is a breeze, by subclassing `TensorType`. For example this checks that the tensor has an additional attribute `foo`, which must be a string with value `"good-foo"`:

```python
from torch import rand, tensor
from torchtyping import TensorType
from typeguard import typechecked

from typing import Any, Optional

# Write the extension

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

# Test the extension

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
```

## Examples

**Shape checking:**

```python
def func(x: TensorType["batch", 5],
         y: TensorType["batch", 3]):
    # x has shape (batch, 5)
    # y has shape (batch, 3)
    # batch dimension is the same for both
	
def func(x: TensorType[2, -1, -1]):
	# x has shape (2, any_one, any_two)
	# -1 is a special value to represent any size.
```

**Checking arbitrary numbers of batch dimensions:**

```python	
def func(x: TensorType[..., 2, 3]):
    # x has shape (..., 2, 3)
	
def func(x: TensorType[..., 2, "channels"],
         y: TensorType[..., "channels"]):
    # x has shape (..., 2, channels)
    # y has shape (..., channels)
    # "channels" is checked to be the same size for both arguments.
	
def func(x: TensorType["batch": ..., "channels_x"],
         y: TensorType["batch": ..., "channels_y"]):
    # x has shape (..., channels_x)
    # y has shape (..., channels_y)
    # the ... batch dimensions are checked to be of the same size.
```

**Return value checking:**

```python
def func(x: TensorType[3, 4]) -> TensorType[()]:
    # x has shape (3, 4)
    # return has shape ()
```

**Dtype checking:**

```python
def func(x: TensorType[float]):
    # x has dtype torch.float32
```

**Checking shape and dtype at the same time:**

```python
def func(x: TensorType[3, 4][float]):
    # x has shape (3, 4) and has dtype torch.float32
```

**Checking names for dimensions as per [named tensors](https://pytorch.org/docs/stable/named_tensor.html):**

```python
def func(x: NamedTensorType["a": 3, "b"]):
    # x has has names ("a", "b")
    # x has shape (3, Any)
```

**Checking layouts:**

```python
def func(x: TensorType[torch.sparse_coo]):
    # x is a sparse tensor with layout sparse_coo
```
