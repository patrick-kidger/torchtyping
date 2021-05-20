import inspect
import sys
import torch
import typeguard

from .tensor_details import _Dim, _no_name, ShapeDetail
from .tensor_type import _AnnotatedType

from typing import Any, Dict, List, Tuple

# get_args is available in python version 3.8
# get_type_hints with include_extras parameter is available in 3.9 PEP 593.
if sys.version_info >= (3, 9):
    from typing import get_type_hints, get_args, Type
else:
    from typing_extensions import get_type_hints, get_args, Type


# TYPEGUARD PATCHER
#######################
# So there's quite a lot of moving pieces here.
# The logic proceeds as follows.
#
# Calling patch_typeguard() just monkey-patches some of its functions and classes.
#
# typeguard uses a `_CallMemo` object to store information about each function that it
# is checking: this is what allows us to perform function-level checking (consistency
# of tensor shapes) rather than just argument-level checking (simple isinstance
# checks).
# So the first thing we do is enhance that with a couple extra slots to store our
# information
#
# Second, we patch `check_type`. typeguard traverses the [] hierarchy, e.g. from
# Tuple[List[int]] to List[int] to int, recursively calling `check_type`. By patching
# `check_type` we can check for our `TensorType`s and record every value-type pair.
# (Actually it's a bit more than that: we record some names for use in the error
# messages.) These are recorded in our enhanced `_CallMemo` object.
#
# (Incidentally we also have to patch typeguard's use of typing.get_type_hints, so that
# our annotations aren't stripped.)
#
# Then we patch `check_argument_types` and `check_return_type`, to perform our extra
# TensorType checking. This is the same checking in both cases so we factor that out
# into _check_memo.
#
# _check_memo performs the real logic of the checking here. This looks at all the
# recorded value-type pairs and checks for any inconsistencies.


def _to_string(name, detail_reprs: List[str]) -> str:
    assert len(detail_reprs) > 0
    string = name + "["
    pieces = []
    for detail_repr in detail_reprs:
        if detail_repr != "":
            pieces.append(detail_repr)
    string += ", ".join(pieces)
    string += "]"
    return string


def _check_tensor(
    argname: str, value: Any, origin: Type[torch.Tensor], metadata: Dict[str, Any]
):
    details = metadata["details"]
    if not isinstance(value, origin) or any(
        not detail.check(value) for detail in details
    ):
        expected_string = _to_string(
            metadata["cls_name"], [repr(detail) for detail in details]
        )
        if isinstance(value, torch.Tensor):
            given_string = _to_string(
                metadata["cls_name"], [detail.tensor_repr(value) for detail in details]
            )
        else:
            value = type(value)
            if hasattr(value, "__qualname__"):
                given_string = value.__qualname__
            elif hasattr(value, "__name__"):
                given_string = value.__name__
            else:
                given_string = repr(value)
        raise TypeError(
            f"{argname} must be of type {expected_string}, got type {given_string} "
            "instead."
        )


def _check_memo(memo):
    ###########
    # Parse the tensors and figure out the sizes of all labelled
    # dimensions.
    # This also performs some (and in practice most) of the consistency
    # checks. However its job is primarily one of assigning sizes to labels.
    # The final checking of the inferred sizes is performed afterwards.
    #
    # This logic is a bit hairy. Most of the complexity comes from
    # supporting `...` arbitrary numbers of dimensions.
    ###########

    # ordered set
    shape_info = {
        (argname, value.shape, detail): None
        for argname, value, _, detail in memo.value_info
    }
    while len(shape_info):
        for argname, shape, detail in shape_info:
            num_free_ellipsis = 0
            for dim in detail.dims:
                if dim.size is ... and dim.name not in memo.name_to_shape:
                    num_free_ellipsis += 1
            if num_free_ellipsis <= 1:
                reversed_shape = enumerate(reversed(shape))
                for dim in reversed(detail.dims):
                    try:
                        reverse_dim_index, size = next(reversed_shape)
                    except StopIteration:
                        if dim.size is ...:
                            if dim.name not in (None, _no_name):
                                try:
                                    lookup_shape = memo.name_to_shape[dim.name]
                                except KeyError:
                                    memo.name_to_shape[dim.name] = ()
                                else:
                                    if lookup_shape != ():
                                        raise TypeError(
                                            f"Dimension group '{dim.name}' of "
                                            f"inconsistent shape. Got both () and "
                                            f"{lookup_shape}."
                                        )
                        else:
                            # I don't think it's possible to get here, as the earlier
                            # call to _check_tensor in check_type should catch
                            # this case.
                            raise TypeError(
                                f"{argname} has {len(shape)} dimensions but type "
                                "requires more than this."
                            )

                    if dim.name not in (None, _no_name):
                        if dim.size is ...:
                            try:
                                lookup_shape = memo.name_to_shape[dim.name]
                            except KeyError:
                                # Can only get here if we're the single free
                                # ellipsis.
                                # Therefore the number of dimensions the ellipsis
                                # corresponds to, is the number of dimensions
                                # remaining.
                                forward_index = 0
                                for forward_dim in detail.dims:  # now iterate forwards
                                    if forward_dim is dim:
                                        break
                                    assert forward_dim.size is ...
                                    forward_index += len(
                                        memo.name_to_shape[forward_dim.name]
                                    )
                                if reverse_dim_index == 0:
                                    # since [:-0] doesn't work
                                    end_index = None
                                else:
                                    end_index = -reverse_dim_index
                                clip_shape = shape[forward_index:end_index]
                                memo.name_to_shape[dim.name] = tuple(clip_shape)
                                for _ in range(len(clip_shape) - 1):
                                    next(reversed_shape)
                            else:
                                reversed_shape_piece = []
                                if len(lookup_shape) >= 1:
                                    reversed_shape_piece.append(size)
                                for _ in range(len(lookup_shape) - 1):
                                    try:
                                        _, size = next(reversed_shape)
                                    except StopIteration:
                                        break
                                    reversed_shape_piece.append(size)

                                shape_piece = tuple(reversed(reversed_shape_piece))
                                if lookup_shape != shape_piece:
                                    raise TypeError(
                                        f"Dimension group '{dim.name}' of "
                                        f"inconsistent shape. Got both {shape_piece} "
                                        f"and {lookup_shape}."
                                    )
                        else:
                            names_to_check = (
                                [dim.name, dim.size]
                                if isinstance(dim.size, str)
                                else [dim.name]
                            )
                            for name in names_to_check:
                                try:
                                    lookup_size = memo.name_to_size[name]
                                except KeyError:
                                    memo.name_to_size[name] = size
                                else:
                                    # Technically not necessary, as one of the
                                    # sizes will override the other, and then the
                                    # instance check will fail.
                                    # This gives a nicer error message though.
                                    if lookup_size != size:
                                        raise TypeError(
                                            f"Dimension '{dim.name}' of inconsistent"
                                            f" size. Got both {size} and "
                                            f"{lookup_size}."
                                        )

                del shape_info[argname, shape, detail]
                break
        else:
            if len(shape_info):
                names = {argname for argname, _, _ in shape_info}
                raise TypeError(
                    f"Could not resolve the size of all `...` in {names}. Either:\n"
                    "(1) the specification is ambiguous. For example "
                    "`func(tensor: TensorType['x': ..., 'y': ...])`.\n"
                    "(2) or repeated named `...` are used without being able to "
                    "resolve the size of those named `...` via another argument "
                    "For example `func(tensor: TensorType['x': ..., 'x': ...])`. "
                    "(But `func(tensor1: TensorType['x': ..., 'x': ...], tensor2: "
                    "TensorType['x': ...])` would be fine.)\n"
                    "\n"
                    "Removing the names of the `...` should suffice to resolve this "
                    "error. (But will of course remove that checking as well.)"
                )

    ###########
    # Do the final checking with the inferred sizes filled in.
    # In practice, malformed inputs will usually trip one of the
    # checks in the previous logic, so this block doesn't actually raise
    # errors very often. (In 1/37 tests at time of writing.)
    # A potential performance improvement might be to integrate it into
    # the previous block.
    ###########

    for argname, value, cls_name, detail in memo.value_info:
        dims = []
        for dim in detail.dims:
            size = dim.size
            if dim.name not in (None, _no_name):
                if size == -1:
                    size = memo.name_to_size[dim.name]
                elif isinstance(size, str):
                    size = memo.name_to_size[size]
                elif size is ...:
                    # This assumes that named Ellipses only occur to the
                    # right of unnamed Ellipses, to avoid filling in
                    # Ellipses that occur to the left of other Ellipses.
                    for size in memo.name_to_shape[dim.name]:
                        dims.append(_Dim(name=_no_name, size=size))
                    continue
            dims.append(_Dim(name=dim.name, size=size))
        detail = detail.update(dims=tuple(dims))
        _check_tensor(
            argname, value, torch.Tensor, {"cls_name": cls_name, "details": [detail]}
        )


unpatched_typeguard = True


def patch_typeguard():
    global unpatched_typeguard
    if unpatched_typeguard:
        unpatched_typeguard = False

        # Defined dynamically, in case something else is doing similar levels of hackery
        # patching typeguard. We want to get typeguard._CallMemo at the time we patch,
        # not any earlier. (Someone might have replaced it since the import statement.)
        class _CallMemo(typeguard._CallMemo):
            __slots__ = (
                "value_info",
                "name_to_size",
                "name_to_shape",
            )
            value_info: List[Tuple[str, torch.Tensor, str, Dict[str, Any]]]
            name_to_size: Dict[str, int]
            name_to_shape: Dict[str, Tuple[int]]

        _check_type = typeguard.check_type
        _check_argument_types = typeguard.check_argument_types
        _check_return_type = typeguard.check_return_type

        check_type_signature = inspect.signature(_check_type)
        check_argument_types_signature = inspect.signature(_check_argument_types)
        check_return_type_signature = inspect.signature(_check_return_type)

        def check_type(*args, **kwargs):
            bound_args = check_type_signature.bind(*args, **kwargs).arguments
            argname = bound_args["argname"]
            value = bound_args["value"]
            expected_type = bound_args["expected_type"]
            memo = bound_args["memo"]
            # First look for an annotated type
            is_torchtyping_annotation = (
                memo is not None
                and hasattr(memo, "value_info")
                and isinstance(expected_type, _AnnotatedType)
            )
            # Now check if it's annotating a tensor
            if is_torchtyping_annotation:
                base_cls, *all_metadata = get_args(expected_type)
                if not issubclass(base_cls, torch.Tensor):
                    is_torchtyping_annotation = False
            # Now check if the annotation's metadata is our metadata
            if is_torchtyping_annotation:
                for metadata in all_metadata:
                    if isinstance(metadata, dict) and "__torchtyping__" in metadata:
                        break
                else:
                    is_torchtyping_annotation = False
            if is_torchtyping_annotation:
                # We call _check_tensor here -- despite calling _check_tensor again
                # once we've seen every argument and filled in the shape details --
                # just because we want to check that `value` is in fact a tensor before
                # we access its `shape` field on the next line.
                _check_tensor(argname, value, base_cls, metadata)
                for detail in metadata["details"]:
                    if isinstance(detail, ShapeDetail):
                        memo.value_info.append(
                            (argname, value, metadata["cls_name"], detail)
                        )
                        break

            else:
                _check_type(*args, **kwargs)

        def check_argument_types(*args, **kwargs):
            bound_args = check_argument_types_signature.bind(*args, **kwargs).arguments
            memo = bound_args["memo"]
            if memo is None:
                return _check_argument_types(*args, **kwargs)
            else:
                memo.value_info = []
                memo.name_to_size = {}
                memo.name_to_shape = {}
                retval = _check_argument_types(*args, **kwargs)
                try:
                    _check_memo(memo)
                except TypeError as exc:  # suppress long traceback
                    raise TypeError(*exc.args) from None
                return retval

        def check_return_type(*args, **kwargs):
            bound_args = check_return_type_signature.bind(*args, **kwargs).arguments
            memo = bound_args["memo"]
            if memo is None:
                return _check_return_type(*args, **kwargs)
            else:
                # Reset the collection of things that need checking.
                memo.value_info = []
                # Do _not_ set memo.name_to_size or memo.name_to_shape, as we want to
                # keep using the same sizes inferred from the arguments.
                retval = _check_return_type(*args, **kwargs)
                try:
                    _check_memo(memo)
                except TypeError as exc:  # suppress long traceback
                    raise TypeError(*exc.args) from None
                return retval

        typeguard._CallMemo = _CallMemo
        typeguard.check_type = check_type
        typeguard.check_argument_types = check_argument_types
        typeguard.check_return_type = check_return_type
        typeguard.get_type_hints = lambda *args, **kwargs: get_type_hints(
            *args, **kwargs, include_extras=True
        )
