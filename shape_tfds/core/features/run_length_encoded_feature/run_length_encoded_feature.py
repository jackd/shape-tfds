"""(Binary) run length encoded ((B)RLE) feature connectors.

Provides `RunLengthEncodedFeature` and `BinaryRunLengthEncodedFeature` for encoding each variant.

See `shape_tfds/core/features/run_length_encoded_feature/README.md` for for details.
"""
import abc
import numpy as np
import six
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core import features
# from tensorflow_datasets.core.features import top_level_feature
from shape_tfds.core.features.run_length_encoded_feature import rle as tf_impl
from trimesh.voxel import runlength as np_impl


def _check_size(data_size, expected_size):
  if expected_size is not None and data_size != expected_size:
    raise ValueError(
        "encoding size %d is incompatible with expected size %d"
        % (data_size, expected_size))

def _check_dtype(dtype, expected_dtype):
  if dtype != expected_dtype:
    raise ValueError(
      "dtype %s does not match expected %s" % (dtype, expected_dtype))


class RunLengthEncodedFeatureBase(features.FeatureConnector):
  _encoded_dtype = tf.int64

  def __init__(self, size, dtype):
    self._size = size
    self._dtype = dtype

  def get_serialized_info(self):
    return tfds.features.TensorInfo(shape=(None,), dtype=self._encoded_dtype)
  
  def get_tensor_info(self):
    return tfds.features.TensorInfo(shape=(self._size,), dtype=self._dtype)
  
  def _shaped(self, decoded):
    if self._size is not None:
      decoded.set_shape((self._size))
    return decoded


class RunLengthEncodedFeature(RunLengthEncodedFeatureBase):
  """`FeatureConnector` for run length encoded 1D tensors."""
  def __init__(self, size=None, dtype=tf.int64):
    super(RunLengthEncodedFeature, self).__init__(size, dtype)

  def decode_example(self, tfexample_data):
    out = self._shaped(tf_impl.rle_to_dense(tfexample_data))
    return tf.cast(out, dtype=self._dtype)

  def encode_example(self, example_data):
    """Encode the given example.

    Args:
      example_data: dense values to encode

    Returns:
      run length encoding int64 array

    Raises:
      `ValueError` if the input size is inconsistent with `size` provided in
      the constructor or the `example_data.dtype` is not int64.
    """
    # only supports dense -> rle encoding
    _check_dtype(example_data.dtype, self._dtype.as_numpy_dtype)
    _check_size(example_data.size, self._size)
    return np_impl.dense_to_rle(
      example_data, dtype=self._encoded_dtype.as_numpy_dtype)


class BinaryRunLengthEncodedFeature(RunLengthEncodedFeatureBase):
  """`FeatureConnector` for binary run length encoded 1D tensors."""
  def __init__(self, size=None):
    super(BinaryRunLengthEncodedFeature, self).__init__(size, tf.bool)

  def decode_example(self, tfexample_data):
    return self._shaped(tf_impl.brle_to_dense(tfexample_data))
  
  def encode_example(self, example_data):
    _check_dtype(example_data.dtype, self._dtype.as_numpy_dtype)
    _check_size(example_data.size, self._size)
    return np_impl.dense_to_brle(
      example_data, dtype=self._encoded_dtype.as_numpy_dtype)
