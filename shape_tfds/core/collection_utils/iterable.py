from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def _first(iterator):
    try:
        return next(iterator)
    except StopIteration:
        raise ValueError('iterable must not be empty to get first element')


def split(iterable):
    """
    Split the head from the rest of the elements in iterable.

    Raises:
        ValueError if iterable is empty
    """
    it = iter(iterable)
    first = _first(it)
    def rest_fn():
        try:
            while True:
                yield next(it)
        except StopIteration:
            pass
    rest = rest_fn()
    return first, rest


def first(iterable):
    """
    Get the first entry from iterable.

    Raises:
        ValueError if iterable is empty
    """
    return _first(iter(iterable))


def single(iterable):
    """
    Get the single entry from iterable.

    Raises:
        ValueError if more than 1 entry found.
    """
    it = iter(iterable)
    first = _first(it)
    try:
        next(it)
        raise ValueError('iterable had more than one entry')
    except StopIteration:
        return first
