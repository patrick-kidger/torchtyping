**0.1.3**

`TensorType` now inherits from `torch.Tensor` so that IDE lookup+error messages work as expected.  
Updated pre-commit hooks. These were failing for some reason.

**0.1.2**

Added support for `str: str` pairs and `None: str` pairs.

**0.1.1**

Added support for Python 3.7+. (Down from Python 3.9+.)  
Added support for `typing.Any` to indicate an arbitrary-size dimension.

**0.1.0**

Initial release.
