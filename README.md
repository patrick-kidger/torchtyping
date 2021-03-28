# torchtyping

Turn this:
```python
def batch_outer_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # x has shape (batch, x_channels)
    # y has shape (batch, y_channels)
    # return has shape (batch, x_channels, y_channels)

    return x.unsqueeze(-1) * y.unsqueeze(-2)
```
into this:
```python
def batch_outer_product(x:   TensorType["batch", "x_channels"],
                        y:   TensorType["batch", "y_channels"]
                        ) -> TensorType["batch", "x_channels", "y_channels"]:

    return x.unsqueeze(-1) * y.unsqueeze(-2)
```
**with programmatic checking that the shape (dtype, ...) specification is met.**

Bye-bye bugs!

---

## Installation

```bash
pip install git+https://github.com/patrick-kidger/torchtyping.git
```

## Details

torchtyping allows for more precisely specifying the details of tensor arguments as part of a type annotation:
- shape: size, number of dimensions
- dtype (float, integer, etc.)
- layout (dense, sparse)
- names of dimensions as per [named tensors](https://pytorch.org/docs/stable/named_tensor.html).
- use `...` to indicate an arbitrary number of batch dimensions
- ...plus anything else you like, as torchtyping is highly extensible.

If [typeguard](https://github.com/agronholm/typeguard) is installed (and being used) then **at runtime the types will be checked** to ensure that the tensors really are of the advertised shape, dtype, etc. (If you're not already using typeguard for your regular Python programming, then strongly consider using it. It's a great way to squash bugs.)

In the example above, then `x`, `y`, and the return value, are all checked to see that their first dimensions (`"batch"`) are the same size as each other. Likewise the `"x_channels"`, `"y_channels"` dimensions are checked against each other.

_Note that to get the programmatic checking, then typeguard must be installed, and activated for the specified functions in its usual way. torchtyping can be used without typeguard -- to clearly document the expected shape, dtype etc. of the the tensors -- but is a lot less useful overall._

## More examples

**Shape checking:**
```python
def func(x: TensorType["batch", 5],
         y: TensorType["batch", 3]):
    # x has shape (batch, 5)
    # y has shape (batch, 3)
    # batch dimension is the same for both
	
def func(x: TensorType[2, -1, -1]):
	# x has shape (2, Any, Any)
```

**Checking arbitrary numbers of batch dimensions:**
```python	
def func(x: TensorType[..., 2, 3]):
    # x has shape (..., 2, 3)
	
def func(x: TensorType[..., 2, "channels"],
         y: TensorType[..., "channels"):
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

## Why is this useful?

I find myself spending a lot of time annotating my code with comments like `# x has shape (batch, hidden_channels)` just to keep track of what shape any given tensor is.

Getting this wrong is an easy way to get silent bugs, when your tensors don't do what you think they do.

Meanwhile, the excellent [typeguard](https://github.com/agronholm/typeguard) package provides a way to check that any argument or return value from a function is actually what its type annotation claims it is. A very useful way to catch bugs.

torchtyping solves the first problem by providing PyTorch types that typeguard can recognise.

Additionally -- this is the real magic -- it performs an additional level of checking over multiple arguments, checking that not only are they of the right type, but that they are collectively of consistent shapes.

## API

```python
torchtyping.TensorType
```
The basic core of the library. Specify shapes, dtypes, dimension names, and even layouts as per the examples above, using the `[]` syntax.

Check multiple things at once by chaining multiple `[]` in any order, for example `torchtyping.TensorType[3, 4][float][torch.strided]`.

```python
torchtyping.NamedTensorType
```
By default the names associated with each dimension in `TensorType` are not checked against the dimension names used as part of being a [named tensor](https://pytorch.org/docs/stable/named_tensor.html).

This is as a convenience, as named tensors aren't used very frequently, and often we still want to name the `TensorType` dimensions to check that they're the same size as other arguments.

`NamedTensorType` performs these additional checks.

```python
torchtyping.FloatTensorType
torchtyping.NamedFloatTensorType
```
There's quite a few floating point types: `torch.float16`, `torch.bfloat16`, `torch.float32`, `torch.float64`, `torch.complex64`, `torch.complex128`. Frequently we're not that fussed which one we get.

These are a convenience to allow any such dtype.

## Custom extensions

Writing custom extensions is a breeze, by subclassing `TensorType`. For example this checks that the tensor has an additional attribute `foo`, which must be a string with value `"good-foo"`:

```python
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
			
			
def foo_checker(x: FooTensorType[FooType("good-foo")]):
    pass
```

See [`extensions.py`](./examples/extensions.py) for more details.