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

Requires Python 3.9+.

## Details

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

patch_typeguard()

@typechecked
def func(x: TensorType["batch"],
         y: TensorType["batch"]) -> TensorType["batch"]:
    return x + y

func(rand(3), rand(3))  # works
func(rand(3), rand(1))
# TypeError: Dimension 'batch' of inconsistent size. Got both 1 and 3.
```

`typeguard` also has an import hook to automatically test an entire module, without needing to add `@typechecked` decorators.

The `patch_typeguard()` call can happen any time before runtime. (And both before or after a `typeguard` import hook is fine.) If you're not using `typeguard` it can be omitted altogether, and `torchtyping` just used for documentation purposes.

If you're not already using `typeguard` for your regular Python programming, then strongly consider using it. It's a great way to squash bugs. Both `typeguard` and `torchtyping` also integrate with `pytest`, so if you're concerned about any performance penalty then they can be enabled during tests only.

## Core API

```python
torchtyping.TensorType[...shape...][...dtype...][...layout...]
```

The core of the library. Specify shapes, dtypes, dimension names, and layouts using the `[]` syntax.

- The shape argument can be any of:
  - An `int`: the dimension must be of exactly this size. If it is `-1` then any size is allowed.
  - A `str`: the size of the dimension passed at runtime will be bound to this name, and all tensors checked that the sizes are consistent.
  - A `...`: An arbitrary number of dimensions.
  - A `str: int` pair (technically it's a slice), combining both `str` and `int` behaviour. (Just a `str` on its own is equivalent to `str: -1`.)
  - A `str: ...` pair, in which case the multiple dimensions corresponding to `...` will be bound to the name specified by `str`, and again checked for consistency between arguments.
  - Any tuple of the above, e.g. `TensorType["batch": ..., "length": 10, "channels", -1]`
- The dtype argument can be any of:
  - `torch.float32`, `torch.float64` etc.
  - `int`, `bool`, `float`, which are converted to their corresponding PyTorch types. `float` is specifically interpreted as `torch.get_default_dtype()`, which is usually `float32`.
- The layout argument can be either `torch.strided` or `torch.sparse_coo`, for dense and sparse tensors respectively.

Check multiple things at once by chaining multiple `[]` in any order. For example `torchtyping.TensorType[3, 4][float][torch.strided]`.

For other things like checking [named tensors](https://pytorch.org/docs/stable/named_tensor.html), see the [further documentation](./FURTHER-DOCUMENTATION.md).

```python
torchtyping.patch_typeguard()
```

`torchtyping` integrates with typeguard to perform runtime type checking. If you want to enable this checking, then make sure to call this function before runtime. It's safe to call before or after the `typeguard` import hook, or after calling `typeguard.typechecked` on a function. It just needs to happen before you actually run the functions you want to check.

This function is safe to run multiple times. Probably the most sensible pattern is to run it once at the top of each file that uses `torchtyping`.

```bash
pytest --torchtyping-patch-typeguard --typeguard-packages="your_package_here"
```

`torchtyping` offers a pytest plugin to automatically run `patch_typeguard` during your tests. Packages can then be passed to typeguard as normal.

## Further documentation

See the [further documentation](./FURTHER-DOCUMENTATION.md) for:

- More examples;
- How to write custom extensions to `torchtyping`;
- Further details of `torchtyping`'s API;
- FAQ.