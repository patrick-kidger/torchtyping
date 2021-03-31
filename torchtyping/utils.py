class frozendict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Calling this immediately ensures that no unhashable types are used as
        # entries.
        # There's also no way this is an efficient hash algorithm, but we're only
        # planning on using this with small dictionaries.
        self._hash = hash(tuple(sorted(self.items())))

    def __setitem__(self, item):
        raise RuntimeError(f"Cannot add items to a {type(self)}.")

    def __delitem__(self, item):
        raise RuntimeError(f"Cannot delete items from a {type(self)}.")

    def __hash__(self):
        return self._hash
