# Further documentation

## Design goals

`torchtyping` had a few design goals.

- **Use type annotations.** There's a few other libraries out there that do this via, essentially, syntactic sugar around `assert` statements. I wanted something neater than that.
- **It should be easy to stop using `torchtyping`.** No, really! If it's not for you then it's easy to remove afterwards. Using `torchtyping` isn't something you should have to bake into your code; just replace `from torchtyping import TensorType` with `TensorType = list` (as a dummy), and your code should still all run.
- **The runtime type checking should be optional.** Runtime checks obviously impose a performance penalty. Integrating with `typeguard` accomplishes this perfectly, in particular through its option to only activate when running tests (my favourite choice).
- **`torchtyping` should be human-readable.** A big part of using type annotations in Python code is to document -- for whoever's reading it -- what is expected. (Particularly valuable on large codebases with several developers.) `torchtyping`'s syntax, and the use of type annotations over some other mechanism, is deliberately chosen to fulfill this goal.

## FAQ

**The runtime checking isn't working!**

First make sure that you're calling `torchtyping.patch_typeguard`.

Then make sure that you've enabled `typeguard`, either by decorating the function with `typeguard.typechecked`, or by using `typeguard.importhook.install_import_hook`, or by using the pytest command line flags listed in the main [README](./README.md).

Make sure that function you're checking is defined _after_ calling `torchtyping.patch_typeguard`.

If you have done all of that, then feel free to raise an issue.

**flake8 is giving spurious warnings.**

Running flake8 will produce spurious warnings for annotations using strings: `TensorType["batch"]` gives `F821 undefined name 'batch'`.

You can silence these en-masse just by creating a dummy `batch = None` anywhere in the file. (Or by placing `# noqa: F821` on the relevant lines.)

**Does this work with `mypy`?**  

Mostly. You'll need to tell `mypy` not to think too hard about `torchtyping`, by annotating its import statements with:

```python
from torchtyping import TensorType  # type: ignore
```

This is because the functionality provided by `torchtyping` is [currently beyond](https://www.python.org/dev/peps/pep-0646/) what `mypy` is capable of representing/understanding. (See also the [links at the end](#other-libraries-and-resources) for further material on this.)

Additionally `mypy` has a bug which causes it crash on any file using the `str: int` or `str: ...` notation, as in `TensorType["batch": 10]`. This can be worked around by skipping the file, by creating a `filename.pyi` file in the same directory. See also the corresponding [mypy issue](https://github.com/python/mypy/issues/10266).

**Are nested annotations of the form `Blahblah[Moreblah[TensorType[...]]]` supported?**

Yes.

**Are multiple `...` supported?**

Yes. For example:

```python
def func(x:  TensorType["dim1": ..., "dim2": ...],
         y:  TensorType["dim2": ...]
        ) -> TensorType["dim1": ...]:
    sum_dims = [-i - 1 for i in range(y.dim())]
    return (x * y).sum(dim=sum_dims)
```

**`TensorType[float]` corresponds to`float32` but `torch.rand(2).to(float)` produces `float64`**.

This is a deliberate asymmetry. `TensorType[float]` corresponds to `torch.get_default_dtype()`, as a convenience, but `.to(float)` always corresponds to `float64`. 

**How to indicate a scalar Tensor, i.e. one with zero dimensions?**

`TensorType[()]`. Equivalently `TensorType[(), float]`, etc.

**Support for TensorFlow/JAX/etc?**

Not at the moment. The library is called `torchtyping` after all. [There are alternatives for these libraries.](#other-libraries-and-resources)

## Custom extensions

Writing custom extensions is a breeze. Checking extra properties is done by subclassing `torchtyping.TensorDetail`, and passing instances of your `detail` to `torchtyping.TensorType`. For example this checks that the tensor has an additional attribute `foo`, which must be a string with value `"good-foo"`:

```python
from torch import rand, Tensor
from torchtyping import TensorDetail, TensorType
from typeguard import typechecked

# Write the extension

class FooDetail(TensorDetail):
    def __init__(self, foo):
        super().__init__()
        self.foo = foo
        
    def check(self, tensor: Tensor) -> bool:
        return hasattr(tensor, "foo") and tensor.foo == self.foo

    # reprs used in error messages when the check is failed
    
    def __repr__(self) -> str:
        return f"FooDetail({self.foo})"

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
```

As you can see, a `detail` must supply three methods. The first is a `check` method, which takes a tensor and checks whether it satisfies the detail. Second is a `__repr__`, which is used in error messages, to describe the detail that wasn't satisfied. Third is a `tensor_repr`, which is also used in error messages, to describe what property the tensor had (instead of the desired detail).

## Other libraries and resources

`torchtyping` is one amongst a few libraries trying to do this kind of thing. Here's some links for the curious:

**Discussion**
- [PEP 646](https://www.python.org/dev/peps/pep-0646/) proposes variadic generics. These are a tool needed for static checkers (like `mypy`) to be able to do the kind of shape checking that `torchtyping` does dynamically. However at time of writing Python doesn't yet support this.
- The [Ideas for array shape typing in Python](https://docs.google.com/document/d/1vpMse4c6DrWH5rq2tQSx3qwP_m_0lyn-Ij4WHqQqRHY/) document is a good overview of some of the ways to type check arrays.

**Other libraries**
- [TensorAnnotations](https://github.com/deepmind/tensor_annotations) is a library for statically checking JAX or TensorFlow tensor shapes. (It also has some good links on to other discussions around this topic.)
- [`nptyping`](https://github.com/ramonhagenaars/nptyping) does something very similar to `torchtyping`, but for numpy.
- [`tsanley`](https://github.com/ofnote/tsanley)/[`tsalib`](https://github.com/ofnote/tsalib) is an alternative dynamic shape checker, but requires a bit of extra setup.
- [TensorGuard](https://github.com/Michedev/tensorguard) is an alternative, using extra function calls rather than type hints.

## More Examples

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
def func(x: TensorType["a": 3, "b", is_named]):
    # x has has names ("a", "b")
    # x has shape (3, Any)
```

**Checking layouts:**

```python
def func(x: TensorType[torch.sparse_coo]):
    # x is a sparse tensor with layout sparse_coo
```
