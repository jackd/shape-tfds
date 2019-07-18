from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import shape_tfds as sds


def BinaryVoxel(shape):
    """
    A feature for encoding binary voxel values.

    Args:
        shape: tuple of ints

    Returns:
        padded shaped binary run length encoded feature
    """
    if not isinstance(shape, tuple):
        raise ValueError('shape must be a tuple, got %s' % str(shape))
    return sds.core.features.PaddedTensor(
        sds.core.features.ShapedTensor(
            sds.core.features.BinaryRunLengthEncodedFeature(),
            (None,)*len(shape)),
        padded_shape=(shape))
