from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections


def _is_base_sequence(value):
    return isinstance(value, (list, tuple, collections.Sequence))


class Sequence(collections.Sequence):

    def map(self, map_fn):
        return MappedSequence(self, map_fn)

    @staticmethod
    def zipped(*args, **kwargs):
        if len(args) == 0:
            return DictSequence(**kwargs)
        elif len(kwargs) == 0:
            return ZippedSequence(*args)
        else:
            raise ValueError('Either args or kwargs must be empty')

    @staticmethod
    def wrapped(sequence):
        return DelegatingSequence(sequence)

    @staticmethod
    def mapped(base, map_fn):
        return MappedSequence(base, map_fn)


class CompoundSequence(Sequence):

    def __init__(self, items):
        for item in items:
            if not _is_base_sequence(item[1]):
                raise TypeError(
                    'all item values must be sequences, but item %s is not' %
                    str(item))
        first, *rest = items
        nf = len(first[1])
        for k, v in rest:
            if len(v) != nf:
                raise ValueError('Inconsistent lengths %s and %s: %d vs %s' %
                                 (first[0], k, nf, len(v)))
        self._len = nf

    def __len__(self):
        return self._len


class DictSequence(CompoundSequence):
    """
    Convert a dict of sequences to a sequence of dicts.

    Example usage:
    ```python
    x = [0, 1, 2, 3]
    y = [10, 11, 12, 13]
    dict_seq = DictSequence(x=x, y=y)
    dict_seq[2] == dict(x=2, y=12)
    len(dict_seq) == 4
    ```
    """

    def __init__(self, **kwargs):
        super(DictSequence, self).__init__(kwargs.items())
        self._children = kwargs

    def __getitem__(self, index):
        return {k: v[index] for k, v in self._children.items()}


class ZippedSequence(CompoundSequence):
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
        super(ZippedSequence, self).__init__(tuple(enumerate(args)))
        self._children = args

    def __getitem__(self, index):
        return tuple(sequence[index] for sequence in self._children)


class DelegatingSequence(Sequence):

    def __init__(self, base):
        if not _is_base_sequence(base):
            raise TypeError('base must be a sequence, got %s' % base)
        self._base = base

    def __len__(self):
        return len(self._base)

    def __getitem__(self, index):
        return self._base[index]


class MappedSequence(DelegatingSequence):
    """
    Lazily mapped sequence.

    Example usage:
    ```python
    x = [0, 2, 5]
    m = MappedSequence(x, lambda x: x**2)
    m[1] == 4
    len(m) == 3
    list(m) == [0, 4, 25]
    ```
    """

    def __init__(self, base, map_fn):
        super(MappedSequence, self).__init__(base)
        self._map_fn = map_fn

    def __getitem__(self, index):
        return self._map_fn(super(MappedSequence, self).__getitem__(index))
