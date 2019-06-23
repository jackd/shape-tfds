from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow_datasets as tfds

# tfds checksums
try:
    import os
    tfds.core.download.checksums._CHECKSUM_DIRS.append(os.path.realpath(
        os.path.join(os.path.dirname(__file__), 'url_checksums')))
    tfds.core.download.checksums._checksum_paths.cache_clear()
except AttributeError:
    # later versions of tfds don't have tfds.core.download.checksums
    # bug seems fixed in these?
    pass


# clean up workspace
del os, tfds

from shape_tfds import core
from shape_tfds import shape

__all__ = [
    'core',
    'shape',
]