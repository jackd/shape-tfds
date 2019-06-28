"""Tests for tensorflow_datasets.core.features.run_length_encoded_feature."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numpy as np
import tensorflow as tf
from tensorflow_datasets.testing import test_utils
from tensorflow_datasets.testing import test_main
import tensorflow_datasets.public_api as tfds
import shape_tfds.core.features as sds_features
tf.compat.v1.enable_eager_execution()

bools = functools.partial(np.array, dtype=np.bool)
ints = functools.partial(np.array, dtype=np.int64)

features = tfds.features


class RunLengthEncodedFeatureTest(test_utils.FeatureExpectationsTestCase):

  def test_brle(self):
    size = 10
    shape = (size,)
    brle_dense_and_encodings = (
        (bools([0, 0, 0, 1, 1, 0, 0, 0, 1, 1]), ints([3, 2, 3, 2])),
        (bools([0, 0, 0, 1, 1, 0, 0, 0, 1, 0]), ints([3, 2, 3, 1, 1])),
    )
    brle_items = []
    for dense, encoding in brle_dense_and_encodings:
      brle_items.append(test_utils.FeatureExpectationItem(
          value=dense,
          expected=dense,
          expected_serialized=list(encoding),
      ))

    self.assertFeature(
        feature=sds_features.BinaryRunLengthEncodedFeature(),
        dtype=tf.bool,
        shape=(None,),
        tests=brle_items
      )
    self.assertFeature(
        feature=sds_features.BinaryRunLengthEncodedFeature(size=size),
        dtype=tf.bool,
        shape=shape,
        tests=brle_items
    )

  def test_rle(self):
    size = 10
    shape = (size,)
    rle_dense_and_encodings = (
        (bools([0, 0, 0, 1, 1, 0, 0, 0, 1, 1]),
         ints([0, 3, 1, 2, 0, 3, 1, 2])),
        (bools([0, 0, 0, 1, 1, 0, 0, 0, 1, 0]),
         ints([0, 3, 1, 2, 0, 3, 1, 1, 0, 1])),
    )
    rle_items = []
    for dense, encoding in rle_dense_and_encodings:
      rle_items.append(test_utils.FeatureExpectationItem(
          value=dense,
          expected=dense,
          expected_serialized=list(encoding),
      ))

    self.assertFeature(
        feature=sds_features.RunLengthEncodedFeature(dtype=tf.bool),
        dtype=tf.bool,
        shape=(None,),
        tests=rle_items
      )
    self.assertFeature(
        feature=sds_features.RunLengthEncodedFeature(
            size=size, dtype=tf.bool),
        dtype=tf.bool,
        shape=shape,
        tests=rle_items
    )


if __name__ == '__main__':
  test_main()
