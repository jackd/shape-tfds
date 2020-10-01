import os

import tensorflow_datasets as tfds

from shape_tfds import core, rendering, shape

checksums_dir = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "url_checksums")
)
try:
    tfds.core.download.add_checksums_dir(checksums_dir)
except Exception:
    # tfds checksums
    try:

        tfds.core.download.checksums._CHECKSUM_DIRS.append(checksums_dir)
        tfds.core.download.checksums._checksum_paths.cache_clear()
    except AttributeError:
        # later versions of tfds don't have tfds.core.download.checksums
        # bug seems fixed in these?
        pass

# clean up workspace
del os, tfds, checksums_dir


__all__ = [
    "core",
    "shape",
    "rendering",
]
