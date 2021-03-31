# Further documentation

## FAQ

**The runtime checking isn't working!**

Make sure that you've enable typeguard, either by decorating the function with `typeguard.typechecked`, or by using `typeguard.importhook.install_import_hook`, or by using the pytest command line flags listed in the main [README](./README.md).

Then make sure that you're calling `torchtyping.patch_typeguard`.

If you have done all of that, then feel free to raise an issue.

**Silencing spurious flake8 warnings.**

Running flake8 will produce spurious warnings for annotations using strings: `TensorType["batch"]` gives `F821 undefined name 'batch'`.

You can silence these en-masse just by creating a dummy `batch = None` anywhere in the file. (Or more laboriously, placing `# noqa: F821` on the relevant lines.)

**Does this work with mypy?**  

Mostly.

The functionality provided by `torchtyping` is [explicitly beyond the current scope of mypy](https://www.python.org/dev/peps/pep-0586/#true-dependent-types-integer-generics), so there's not much hope of tight integration.

But at the very least, `torchtyping` generally shouldn't break mypy. Put a `# type: ignore` comment on the same line in which you import `torchtyping` and you'll be good to go.

There is one exception: using `TensorType["string": value]` hits a bug in mypy and causes a crash. See the corresponding [mypy issue](https://github.com/python/mypy/issues/10266).

**How to indicate a scalar Tensor, i.e. one with zero dimensions?**

`TensorType[()]`. Equivalently `TensorType[(), float]`, etc.

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

Using `...` in a shape specification is currently only supported in the left-most places of a tensor shape. Supporting using `...` in other locations would be a fair bit more complicated to write the logic for.

**`TensorType[float]` corresponds to`float32` but `torch.rand(2).to(float)` produces `float64`**.

This is a deliberate asymmetry. `TensorType[float]` corresponds to `torch.get_default_dtype()`, as a convenience, but `.to(float)` always corresponds to `float64`. 

**Why is `TensorType` implemented the way it is / shouldn't it be implemented in a different way / why use `typeguard` rather than a different package / etc. ?**

The short answer is that this works around the various limitations elsewhere in the ecosystem.

Python does provide a comprehensive typing system that we could have used instead -- `typing.Annotated[torch.Tensor, ...]`,  would probably make the most sense -- but then there's no easy way to add the extra shape checking that gives `torchtyping` its _raison d'etre_.

## Custom extensions

Writing custom extensions is a breeze, by subclassing `torchtyping.TensorDetail`. For example this checks that the tensor has an additional attribute `foo`, which must be a string with value `"good-foo"`:

```python
from torch import rand, Tensor
from torchtyping import TensorDetail, TensorType
from typeguard import typechecked

# Write the extension

class FooDetail(TensorDetail):
    def __init__(self, *, value, **kwargs):
        super().__init__(**kwargs)
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
def foo_checker(tensor: TensorType[float, FooDetail(value="good-foo")]):
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
def func(x: TensorType[3, 4, float]):
    # x has shape (3, 4) and has dtype torch.float32
```

**Checking names for dimensions as per [named tensors](https://pytorch.org/docs/stable/named_tensor.html):**

```python
def func(x: TensorType["a": 3, "b", named_detail]):
    # x has has names ("a", "b")
    # x has shape (3, Any)
```

**Checking layouts:**

```python
def func(x: TensorType[torch.sparse_coo]):
    # x is a sparse tensor with layout sparse_coo
```
