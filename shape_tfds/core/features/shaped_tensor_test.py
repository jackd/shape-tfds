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


class ShapedTensorTest(test_utils.FeatureExpectationsTestCase):

  def test_shape_static(self):

    np_input = np.random.rand(2, 3).astype(np.float32)
    array_input = [
        [1, 2, 3],
        [4, 5, 6],
    ]

    feature = sds_features.ShapedTensor(
        features.Tensor(shape=(6,), dtype=tf.float32), shape=(2, 3))
    self.assertIsInstance(feature, sds_features.StaticShapedTensor)
    self.assertFeature(
        feature=feature,
        dtype=tf.float32,
        shape=(2, 3),
        tests=[
            # Np array
            test_utils.FeatureExpectationItem(
                value=np_input,
                expected=np_input,
            ),
            # Python array
            test_utils.FeatureExpectationItem(
                value=array_input,
                expected=array_input,
            ),
            # Invalid shape
            test_utils.FeatureExpectationItem(
                value=np.random.rand(2, 4).astype(np.float32),
                raise_cls=ValueError,
                raise_msg='cannot reshape',
            ),
        ],
    )

    np_input_dynamic_1 = np.random.randint(256, size=(2, 3, 2), dtype=np.int32)
    np_input_dynamic_2 = np.random.randint(256, size=(5, 3, 2), dtype=np.int32)
    feature = sds_features.ShapedTensor(
        features.Tensor(shape=(None,), dtype=tf.int32), shape=(None, 3, 2))
    self.assertIsInstance(feature, sds_features.StaticShapedTensor)
    self.assertFeature(
        feature=feature,
        dtype=tf.int32,
        shape=(None, 3, 2),
        tests=[
            test_utils.FeatureExpectationItem(
                value=np_input_dynamic_1,
                expected=np_input_dynamic_1,
                expected_serialized=np_input_dynamic_1.flatten()
            ),
            test_utils.FeatureExpectationItem(
                value=np_input_dynamic_2,
                expected=np_input_dynamic_2,
                expected_serialized=np_input_dynamic_2.flatten()
            )
        ]
    )

#   def test_shape_dynamic(self):

#     np_input_dynamic_1 = np.random.randint(256, size=(2, 3, 2), dtype=np.int32)
#     np_input_dynamic_2 = np.random.randint(256, size=(5, 3, 2), dtype=np.int32)
#     feature = sds_features.ShapedTensor(
#         features.Tensor(shape=(None,), dtype=tf.int32),
#         shape=(None, None, 2))
#     self.assertIsInstance(feature, sds_features.DynamicShapedTensor)

#     self.assertFeature(
#         feature=feature,
#         dtype=tf.int32,
#         shape=(None, None, 2),
#         tests=[
#             test_utils.FeatureExpectationItem(
#                 value=np_input_dynamic_1,
#                 expected=np_input_dynamic_1,
#                 expected_serialized=dict(
#                   shape=np_input_dynamic_1.shape,
#                   flat_values=np_input_dynamic_1.flatten()
#                 )
#             ),
#             test_utils.FeatureExpectationItem(
#                 value=np_input_dynamic_2,
#                 expected=np_input_dynamic_2,
#                 expected_serialized=dict(
#                   shape=np_input_dynamic_2.shape,
#                   flat_values=np_input_dynamic_2.flatten()
#                 )
#             ),
#             # Invalid shape
#             test_utils.FeatureExpectationItem(
#                 value=
#                 np.random.randint(256, size=(2, 3, 1), dtype=np.int32),
#                 raise_cls=ValueError,
#                 raise_msg='is incompatible',
#             ),
#         ]
#     )


if __name__ == '__main__':
  test_main()
