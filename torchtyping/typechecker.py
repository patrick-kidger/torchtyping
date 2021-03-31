import inspect
import torch
import typeguard

from .tensor_details import _Dim, ShapeDetail
from .tensor_type import _AnnotatedType

from typing import Any, get_args, Type
from typing import get_type_hints


def _to_string(name, detail_reprs: list[str]) -> str:
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
    argname: str, value: Any, origin: Type[torch.Tensor], metadata: dict[str, Any]
):
    details = metadata["details"]
    if not isinstance(value, origin) or any(
        not detail.check(value) for detail in details
    ):
        expected_string = _to_string(
            metadata["name"], [repr(detail) for detail in details]
        )
        if isinstance(value, torch.Tensor):
            given_string = _to_string(
                metadata["name"], [detail.tensor_repr(value) for detail in details]
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


def _handle_ellipsis_size(
    name: str, name_to_shape: dict[str, tuple[int]], shape: tuple[int]
):
    try:
        lookup_shape = name_to_shape[name]
    except KeyError:
        name_to_shape[name] = shape
    else:
        if lookup_shape != shape:
            raise TypeError(
                f"Dimension group '{name}' of inconsistent shape. Got "
                f"both {shape} and {lookup_shape}."
            )


def _check_memo(memo):
    ###########
    # Parse the tensors and figure out the extra hinting
    ###########

    # ordered set
    shape_info = {
        (argname, value.shape, detail): None
        for argname, value, name, detail in memo.value_info
    }
    while len(shape_info):
        for argname, shape, detail in shape_info:
            num_free_ellipsis = 0
            for dim in detail.dims:
                if dim.size is ... and dim.name not in memo.name_to_shape:
                    num_free_ellipsis += 1
            if num_free_ellipsis <= 1:
                reversed_shape = reversed(shape)
                # These aren't all necessarily of the same size.
                for reverse_dim_index, dim in enumerate(reversed(detail.dims)):
                    try:
                        size = next(reversed_shape)
                    except StopIteration:
                        if dim.size is ...:
                            if dim.name is not None:
                                _handle_ellipsis_size(dim.name, memo.name_to_shape, ())
                        else:
                            raise TypeError(
                                f"{argname} has {len(shape)} dimensions but type "
                                "requires more than this."
                            )

                    if dim.name is not None:
                        if dim.size == -1:
                            try:
                                lookup_size = memo.name_to_size[dim.name]
                            except KeyError:
                                memo.name_to_size[dim.name] = size
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
                        elif dim.size is ...:
                            if reverse_dim_index == 0:
                                # since [:-0] doesn't work
                                clip_shape = shape
                            else:
                                clip_shape = shape[:-reverse_dim_index]
                            _handle_ellipsis_size(
                                dim.name, memo.name_to_shape, clip_shape
                            )
                            # Can only get here if we're the single free
                            # ellipsis.
                            # Therefore the number of dimensions it
                            # corresponds to is the number of dimensions
                            # remaining.
                            break

                        # else (dim.size an integer) branch not included. We
                        # don't check if dim.size == size here, that's done in
                        # the instance check. Here we're just concerned with
                        # resolving the size of names.

                del shape_info[argname, shape, detail]
                break
        else:
            if len(shape_info):
                names = {argname for argname, _, _ in shape_info}
                raise TypeError(
                    f"Could not resolve the size of all `...` in arguments {names}."
                )

    ###########
    # Do the extra checking with the extra tensor details filled in.
    ###########

    for argname, value, name, detail in memo.value_info:
        dims = []
        for dim in detail.dims:
            name = dim.name
            size = dim.size
            if name is not None:
                if size == -1:
                    size = memo.name_to_size[name]
                elif size is ...:
                    # This assumes that named Ellipses only occur to the
                    # right of unnamed Ellipses, to avoid filling in
                    # Ellipses that occur to the left of other Ellipses.
                    for size in memo.name_to_shape[name]:
                        dims.append(_Dim(name=None, size=size))
                    continue
            dims.append(_Dim(name=name, size=size))
        detail = detail.update(dims=tuple(dims))
        _check_tensor(argname, value, torch.Tensor, {"name": name, "details": [detail]})


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
            value_info: list[tuple[str, torch.Tensor, dict[str, Any]]]
            name_to_size: dict[str, int]
            name_to_shape: dict[str, tuple[int]]

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
                # We perform a check here -- despite having the additional more
                # fully-fledged checks below -- for two reasons.
                # One: we want to check return values as well. The argument checking
                # below doesn't cover this.
                # Two: we want to check that `value` is in fact a tensor before we
                # access its `shape` field on the next line.
                _check_tensor(argname, value, base_cls, metadata)
                for detail in metadata["details"]:
                    if isinstance(detail, ShapeDetail):
                        memo.value_info.append(
                            (argname, value, metadata["name"], detail)
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
                _check_memo(memo)
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
                _check_memo(memo)
                return retval

        typeguard._CallMemo = _CallMemo
        typeguard.check_type = check_type
        typeguard.check_argument_types = check_argument_types
        typeguard.check_return_type = check_return_type
        typeguard.get_type_hints = lambda *args, **kwargs: get_type_hints(
            *args, **kwargs, include_extras=True
        )
