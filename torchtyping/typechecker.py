import inspect
import torch
import typeguard

from .tensor_type import TensorType, _Dim


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
                f"Dimension group {name} of inconsistent shape. Got "
                f"both {shape} and {lookup_shape}."
            )


def _check_tensor(argname: str, value, hint):
    if not isinstance(value, hint):
        if isinstance(argname, torch.Tensor):
            like = hint.like(value)
        else:
            like = type(value)
            if hasattr(like, "__qualname__"):
                like = like.__qualname__
        raise TypeError(f"{argname} must be {hint}, got {like} instead.")


def _check_memo(memo):
    # Parse the tensors and figure out the extra hinting
    while len(memo.argname_shape_hints):
        for argname, shape, hint in memo.argname_shape_hints:
            num_free_ellipsis = 0
            for dim in hint.dims:
                if dim.size is ... and dim.name not in memo.name_to_shape:
                    num_free_ellipsis += 1
            if num_free_ellipsis <= 1:
                reversed_shape = reversed(shape)
                # These aren't all necessarily of the same size.
                for reverse_dim_index, dim in enumerate(reversed(hint.dims)):
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
                                        f"Dimension {dim.name} of inconsistent"
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

                memo.argname_shape_hints.remove((argname, shape, hint))
                break
        else:
            if len(memo.argname_shape_hints):
                names = {argname for argname, _, _ in memo.argname_shape_hints}
                raise TypeError(
                    f"Could not resolve the size of all `...` in arguments {names}."
                )

    # Do the extra checking with the extra tensor details filled in.
    for argname, value, hint in memo.argname_value_hints:
        dims = []
        for dim in hint.dims:
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
        hint = hint.update(tuple(dims))
        _check_tensor(argname, value, hint)


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
                "argname_shape_hints",
                "argname_value_hints",
                "name_to_size",
                "name_to_shape",
            )

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
            is_tensor_type = (
                memo is not None
                and hasattr(memo, "argname_shape_hints")
                and inspect.isclass(expected_type)
                and issubclass(expected_type, TensorType)
                and expected_type.dims is not None
            )

            if is_tensor_type:
                # We perform a check here -- despite having the additional more
                # fully-fledged checks below -- for two reasons.
                # One: we want to check return values as well. The argument checking
                # below doesn't cover this.
                # Two: we want to check that `value` is in fact a tensor before we
                # access its `shape` field on the next line.
                _check_tensor(argname, value, expected_type)
                # We store just the shape-type pairs separately. (Even though they can
                # be reconsituted from argname_value_hints.) This is because both
                # shapes and hints can easily have a lot of repetition, so this reduces
                # the set of things we need to check.
                memo.argname_shape_hints.add((argname, value.shape, expected_type))
                memo.argname_value_hints.append((argname, value, expected_type))
            else:
                _check_type(*args, **kwargs)

        def check_argument_types(*args, **kwargs):
            bound_args = check_argument_types_signature.bind(*args, **kwargs).arguments
            memo = bound_args["memo"]
            if memo is None:
                return _check_argument_types(*args, **kwargs)
            else:
                memo.argname_shape_hints = set()
                memo.argname_value_hints = []
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
                # Reset the collections of things that need checking.
                memo.argname_shape_hints = set()
                memo.argname_value_hints = []
                # Do _not_ set memo.name_to_size or memo.name_to_shape, as we want to
                # keep using the same sizes inferred from the arguments.
                retval = _check_return_type(*args, **kwargs)
                _check_memo(memo)
                return retval

        typeguard._CallMemo = _CallMemo
        typeguard.check_type = check_type
        typeguard.check_argument_types = check_argument_types
        typeguard.check_return_type = check_return_type
