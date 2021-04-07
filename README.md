<h1 align='center'>torchtyping</h1>
<h2 align='center'>Type annotations for a tensor's shape, dtype, names, ...</h2>

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

Bye-bye bugs! Say hello to enforced, clear documentation of your code.

If (like me) you find yourself littering your code with comments like `# x has shape (batch, hidden_state)` or statements like `assert x.shape == y.shape` , just to keep track of what shape everything is, **then this is for you.**

---

## Installation

```bash
pip install git+https://github.com/patrick-kidger/torchtyping.git
```

Requires Python 3.9+ and PyTorch 1.7.0+.

## Usage

`torchtyping` allows for type annotating:

- **shape**: size, number of dimensions;
- **dtype** (float, integer, etc.);
- **layout** (dense, sparse);
- **names** of dimensions as per [named tensors](https://pytorch.org/docs/stable/named_tensor.html);
- **arbitrary number of batch dimensions** with `...`;
- **...plus anything else you like**, as `torchtyping` is highly extensible.

If [`typeguard`](https://github.com/agronholm/typeguard) is (optionally) installed then **at runtime the types can be checked** to ensure that the tensors really are of the advertised shape, dtype, etc. 

```python
# EXAMPLE

from torch import rand
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()  # use before @typechecked

@typechecked
def func(x: TensorType["batch"],
         y: TensorType["batch"]) -> TensorType["batch"]:
    return x + y

func(rand(3), rand(3))  # works
func(rand(3), rand(1))
# TypeError: Dimension 'batch' of inconsistent size. Got both 1 and 3.
```

`typeguard` also has an import hook that can be used to automatically test an entire module, without needing to manually add `@typeguard.typechecked` decorators.

If you're not using `typeguard` then `torchtyping.patch_typeguard()` can be omitted altogether, and `torchtyping` just used for documentation purposes. If you're not already using `typeguard` for your regular Python programming, then strongly consider using it. It's a great way to squash bugs. Both `typeguard` and `torchtyping` also integrate with `pytest`, so if you're concerned about any performance penalty then they can be enabled during tests only.

## API

```python
torchtyping.TensorType[shape, dtype, layout, details]
```

The core of the library.

Each of `shape`, `dtype`, `layout`, `details` are optional.

- The `shape` argument can be any of:
  - An `int`: the dimension must be of exactly this size. If it is `-1` then any size is allowed.
  - A `str`: the size of the dimension passed at runtime will be bound to this name, and all tensors checked that the sizes are consistent.
  - A `...`: An arbitrary number of dimensions of any sizes.
  - A `str: int` pair (technically it's a slice), combining both `str` and `int` behaviour. (Just a `str` on its own is equivalent to `str: -1`.)
  - A `str: ...` pair, in which case the multiple dimensions corresponding to `...` will be bound to the name specified by `str`, and again checked for consistency between arguments.
  - `None`, which when used in conjunction with `is_named` below, indicates a dimension that must _not_ have a name in the sense of [named tensors](https://pytorch.org/docs/stable/named_tensor.html).
  - A `None: int` pair, combining both `None` and `int` behaviour. (Just a `None` on its own is equivalent to `None: -1`.)
  - A `typing.Any`: Any size is allowed for this dimension (equivalent to `-1`).
  - Any tuple of the above. For example.`TensorType["batch": ..., "length": 10, "channels", -1]`. If you just want to specify the number of dimensions then use for example `TensorType[-1, -1, -1]` for a three-dimensional tensor.
- The `dtype` argument can be any of:
  - `torch.float32`, `torch.float64` etc.
  - `int`, `bool`, `float`, which are converted to their corresponding PyTorch types. `float` is specifically interpreted as `torch.get_default_dtype()`, which is usually `float32`.
- The `layout` argument can be either `torch.strided` or `torch.sparse_coo`, for dense and sparse tensors respectively.
- The `details` argument offers a way to pass an arbitrary number of additional flags that customise and extend `torchtyping`. Two flags are built-in by default. `torchtyping.is_named` causes the [names of tensor dimensions](https://pytorch.org/docs/stable/named_tensor.html) to be checked, and `torchtyping.is_float` can be used to check that arbitrary floating point types are passed in. (Rather than just a specific one as with e.g. `TensorType[torch.float32]`.) For discussion on how to customise `torchtyping` with your own `details`, see the [further documentation](https://github.com/patrick-kidger/torchtyping/FURTHER-DOCUMENTATION.md#custom-extensions).

Check multiple things at once by just putting them all together inside a single `[]`. For example `TensorType["batch": ..., "length", "channels", float, is_named]`.

```python
torchtyping.patch_typeguard()
```

`torchtyping` integrates with `typeguard` to perform runtime type checking. `torchtyping.patch_typeguard()` should be called at the global level, and will patch `typeguard` to check `TensorType`s.

This function is safe to run multiple times. (It does nothing after the first run). 

- If using `@typeguard.typechecked`, then `torchtyping.patch_typeguard()` should be called any time before using `@typeguard.typechecked`. For example you could call it at the start of each file using `torchtyping`.
- If using `typeguard.importhook.install_import_hook`, then `torchtyping.patch_typeguard()` should be called any time before defining the functions you want checked. For example you could call `torchtyping.patch_typeguard()` just once, at the same time as the `typeguard` import hook. (The order of the hook and the patch doesn't matter.)
- If you're not using `typeguard` then `torchtyping.patch_typeguard()` can be omitted altogether, and `torchtyping` just used for documentation purposes.

```bash
pytest --torchtyping-patch-typeguard
```

`torchtyping` offers a `pytest` plugin to automatically run `torchtyping.patch_typeguard()` before your tests. `pytest` will automatically discover the plugin, you just need to pass the `--torchtyping-patch-typeguard` flag to enable it. Packages can then be passed to `typeguard` as normal, either by using `@typeguard.typechecked`, `typeguard`'s import hook, or the `pytest` flag `--typeguard-packages="your_package_here"`.

## Further documentation

See the [further documentation](https://github.com/patrick-kidger/torchtyping/FURTHER-DOCUMENTATION.md) for:

- FAQ;
  - Including `flake8` and `mypy` compatibility;
- How to write custom extensions to `torchtyping`;
- Resources and links to other libraries and materials on this topic;
- More examples.
