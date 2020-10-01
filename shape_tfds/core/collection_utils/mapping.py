import collections


def _is_base_mapping(value):
    return isinstance(value, collections.Mapping)


class Mapping(collections.Mapping):

    def map(self, map_fn):
        return MappedMapping(self, map_fn)

    def item_map(self, map_fn):
        return ItemMappedMapping(self, map_fn)

    @staticmethod
    def zipped(*args, **kwargs):
        if len(args) == 0:
            return DictMapping(**kwargs)
        elif len(kwargs) == 0:
            return ZippedMapping(*args)
        else:
            raise ValueError('Either args or kwargs must be empty')

    @staticmethod
    def wrapped(mapping):
        return DelegatingMapping(mapping)

    @staticmethod
    def mapped(base, map_fn):
        if _is_base_mapping(base):
            return MappedMapping(base, map_fn)
        elif hasattr(base, '__iter__'):
            return LazyMapping(set(base), map_fn)
        else:
            raise TypeError('base must be a Mapping or iterable, got %s' % base)

    @staticmethod
    def item_mapped(base, map_fn):
        return ItemMappedMapping(base, map_fn)


class DelegatingMapping(Mapping):

    def __init__(self, base):
        self._base = base

    def __len__(self):
        return len(self._base)

    def __getitem__(self, key):
        return self._base[key]

    def __iter__(self):
        return iter(self._base)

    def __contains__(self, key):
        return key in self._base


def _check_items(items):
    for item in items:
        if not _is_base_mapping(item[1]):
            raise TypeError(
                'all item values must be mappings, but item %s is not' %
                str(item))
    first, *rest = items
    nf = len(first[1])
    for k, v in rest:
        if len(v) != nf:
            raise ValueError('Inconsistent lengths %s and %s: %d vs %s' %
                             (first[0], k, nf, len(v)))
    return first[1]


class CompoundMapping(DelegatingMapping):

    def __init__(self, items):
        base = _check_items(items)
        super(CompoundMapping, self).__init__(base)


class DictMapping(CompoundMapping):
    """
    Convert a dict of mappings to a mapping of mappings.

    Example usage:
    ```python
    x = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    y = {'a': 10, 'b': 11, 'c': 12, 'd': 13}
    dict_map = DictMapping(x=x, y=y)
    dict_map['c'] == dict(x=2, y=12)
    len(dict_map) == 4
    ```
    """

    def __init__(self, **kwargs):
        super(DictMapping, self).__init__(kwargs.items())
        self._children = kwargs

    def __getitem__(self, key):
        return {k: v[key] for k, v in self._children.items()}


class FlatMapping(Mapping):

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if not _is_base_mapping(v):
                raise ValueError('expected Mapping for kwarg %s, got %s' %
                                 (k, v))
        self._children = kwargs

    def __len__(self):
        return sum(len(v) for v in self._children.values())

    def _split(self, key):
        if len(key) != 2:
            raise KeyError('Expected key of length 2, got %d' % len(key))
        return key

    def _merge(self, k0, k1):
        return k0, k1

    def __getitem__(self, key):
        k0, k1 = self._split(key)
        return self._children[k0][k1]

    def __iter__(self):
        return iter(self.keys())

    def keys(self):
        for k0, v in self._children.items():
            for k1 in v:
                yield self._merge(k0, k1)

    def __contains__(self, key):
        try:
            k0, k1 = self._split(key)
        except KeyError:
            return False
        return k0 in self._children and k1 in self._children[k0]


class NestedFlatMapping(FlatMapping):

    def _split(self, key):
        return key[0], key[1:]

    def _merge(self, k0, k1):
        return (k0,) + k1


def flat_mapping(depth=1, **kwargs):
    if depth < 1:
        raise ValueError('depth must be at least 1')
    elif depth == 1:
        return DelegatingMapping(kwargs)
    elif depth == 2:
        return FlatMapping(**kwargs)
    else:
        return NestedFlatMapping(
            **{
                k: v if isinstance(v, FlatMapping
                                  ) else flat_mapping(depth=depth - 1, **v)
                for k, v in kwargs.items()
            })


class ZippedMapping(CompoundMapping):
    """
    Sequence interface for zipped sequences.

    Example usage:
    ```python
    x = [0, 1, 2, 3]
    y = [10, 11, 12, 13]
    zipped = ZippedSequence(x, y)
    zipped[2] == (2, 12)
    len(zipped) == 4
    ```
    """

    def __init__(self, *args):
        super(ZippedMapping, self).__init__(tuple(enumerate(args)))
        self._children = args

    def __getitem__(self, key):
        return tuple(child[key] for child in self._children)


class MappedMapping(DelegatingMapping):
    """
    Lazily mapped Mapping.

    Example usage:
    ```python
    x = {'a': 0, 'b': 2, 'c': 5}
    m = MappedMapping(x, lambda x: x**2)
    m['b'] == 4
    len(m) == 3
    dict(m.items()) == {'a': 0, 'b': 4, 'c': 25}
    ```
    """

    def __init__(self, base, map_fn):
        super(MappedMapping, self).__init__(base)
        self._map_fn = map_fn

    def __getitem__(self, key):
        return self._map_fn(super(MappedMapping, self).__getitem__(key))


class ItemMappedMapping(DelegatingMapping):
    """Lazily mapped Mapping with function that takes (key, value) -> value."""

    def __init__(self, base, map_fn):
        super(ItemMappedMapping, self).__init__(base)
        self._map_fn = map_fn

    def __getitem__(self, key):
        return self._map_fn(key,
                            super(ItemMappedMapping, self).__getitem__(key))


class LazyMapping(Mapping):

    def __init__(self, keys, map_fn):
        self._keys = keys
        self._map_fn = map_fn

    def __getitem__(self, key):
        if key not in self:
            raise KeyError('invalid key %s' % key)
        return self._map_fn(key)

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

    def __contains__(self, key):
        return key in self._keys
