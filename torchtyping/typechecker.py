import typeguard


_typechecked = typeguard.typechecked


def typechecked(*args, **kwargs):
    pass


patched_typeguard = False


def patch_typeguard():
    if not patched_typeguard:
        global patched_typeguard
        patched_typeguard = True
        typeguard.typechecked = typechecked
