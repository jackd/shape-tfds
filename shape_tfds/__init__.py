import os

import tensorflow_datasets as tfds

from shape_tfds import core, rendering, shape

checksums_dir = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "url_checksums")
)
version = tfds.__version__  # pylint: disable=no-member
if version < "4.0.2" and version != "4.0.1+nightly":
    raise ImportError("Requires tensorflow_datasets >= 4.0.2 or tfds-nightly >= 4.0.1")

tfds.core.download.add_checksums_dir(checksums_dir)
# clean up workspace
del os, tfds, checksums_dir, version


__all__ = [
    "core",
    "shape",
    "rendering",
]
