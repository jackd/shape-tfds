from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import contextlib
import zlib


def get_random_seed(key, seed_offset=0):
    """Get a deterministic seed from key and seed_offset."""
    return (zlib.adler32(str.encode(key)) + seed_offset) % (2**32)


def get_random_state(key, seed_offset=0):
    """Get an np.random.RandomState seeded by `get_random_seed`."""
    seed = get_random_seed(key, seed_offset)
    return np.random.RandomState(seed)  # pylint: disable=no-member


@contextlib.contextmanager
def random_context(random):
    """
    Temporarily use random.get_state in np.random inside a context block.

    Example usage:
    ```python
    state0 = np.random.RandomState(0)
    r0 = state0.uniform()

    state0 = np.random.RandomState(0)
    with random_context(state0):
        r = np.random.uniform()
        assert(r == r0)

    np.random.set_state(np.random.RandomState(0).get_state())
    with random_context(np.random.RandomState(1)):
        np.random.uniform()
    r = np.random.uniform()
    assert(r == r0)
    ```
    """
    base_state = np.random.get_state()
    np.random.set_state(random.get_state())
    yield None
    np.random.set_state(base_state)
