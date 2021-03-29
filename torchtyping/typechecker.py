import typeguard

from .tensor_type import TensorType, _Dim


def _handle_ellipsis_size(name: str, detail: dict[str, int], num_dims: int):
    try:
        detail_num_dims = detail[name]
    except KeyError:
        detail[name] = num_dims
        return True
    else:
        if detail_num_dims != num_dims:
            raise TypeError(
                f"Dimension group {name} of inconsistent number of dimensions. Got "
                "both {num_dims} and {detail_num_dims}."
            )
        return False


unpatched_typeguard = True


def patch_typeguard():
    global unpatched_typeguard
    if unpatched_typeguard:
        unpatched_typeguard = False

        # Defined dynamically, in case something else is doing similar levels of hackery
        # patching typeguard. We want to get typeguard._CallMemo at the time we patch,
        # not any earlier. (In case someone has replaced it since the import statement.)
        class _CallMemo(typeguard._CallMemo):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

                detail = {}
                tensor_type_hints = {
                    name: hint
                    for name, hint in self.type_hints.items()
                    if issubclass(hint, TensorType) and hint.dims is not None
                }
                tensor_type_hints_backup = tensor_type_hints.copy()

                # O(n^2), hope you don't have too many arguments using `...`.
                # Would need to construct a dependency graph between arguments to do
                # this faster, but realistically this shouldn't be an issue. How many
                # `...` can one argument need?
                while len(tensor_type_hints):
                    for name, hint in tensor_type_hints.items():
                        num_free_ellipsis = 0
                        for dim in hint.dims:
                            if dim.size is ... and dim.name not in detail:
                                num_free_ellipsis += 1
                        if num_free_ellipsis > 1:
                            continue

                        shape = self.arguments[name].shape
                        reversed_shape = reversed(shape)
                        # These aren't all necessarily of the same size.
                        for reverse_dim_index, dim in enumerate(reversed(hint.dims)):
                            try:
                                size = next(reversed_shape)
                            except StopIteration:
                                if dim.size is ...:
                                    if dim.name is not None:
                                        _handle_ellipsis_size(dim.name, detail, 0)
                                else:
                                    raise TypeError(
                                        f"{name} has {len(shape)} dimensions but type "
                                        "requires more than this."
                                    )

                            if dim.name is not None:
                                if dim.size == -1:
                                    try:
                                        detail_size = detail[dim.name]
                                    except KeyError:
                                        detail[dim.name] = size
                                    else:
                                        # Technically not necessary, as one of the
                                        # sizes will override the other, and then the
                                        # instance check will fail.
                                        # This gives a nicer error message though.
                                        if detail_size != size:
                                            raise TypeError(
                                                f"Dimension {dim.name} of inconsistent"
                                                " size. Got both {size} and "
                                                "{detail_size}."
                                            )
                                elif dim.size is ...:
                                    num_dims = len(shape) - reverse_dim_index
                                    new_ellipsis = _handle_ellipsis_size(
                                        dim.name, detail, num_dims
                                    )
                                    if new_ellipsis:
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

                        del tensor_type_hints[name]
                    else:
                        if len(tensor_type_hints):
                            raise TypeError("Could not resolve the size of all `...`.")

                for name, hint in tensor_type_hints_backup.items():
                    dims = []
                    for dim in hint.dims:
                        name = dim.name
                        size = dim.size
                        num_dims = 1
                        if name is not None:
                            if size == -1:
                                size = detail[name]
                            if size is ...:
                                # This assumes that named Ellipses only occur to the
                                # right of unnamed Ellipses, to avoid filling in
                                # Ellipses that occur to the left of other Ellipses.
                                size = -1
                                num_dims = detail[name]
                        for _ in range(num_dims):
                            dims.append(_Dim(name=name, size=size))
                    self.type_hints[name] = hint.update(tuple(dims))

        typeguard._CallMemo = _CallMemo
