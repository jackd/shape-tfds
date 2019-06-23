from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

from tensorflow_datasets.core import features

class PaddedTensor(features.feature.FeatureConnector):
  def __init__(self, stripped_feature, pad_kwargs=None, strip_fn=None):
    self._stripped_feature = stripped_feature
    stripped_info = stripped_feature.get_tensor_info()
    self._num_dims = len(stripped_info.shape)
    self._dtype = stripped_info.dtype
    self._padding_feature = features.feature.Tensor(
        shape=(self._num_dims, 2), dtype=tf.int64)
    self._pad_kwargs = {} if pad_kwargs is None else pad_kwargs
    self._strip_fn = strip_fn

  def get_tensor_info(self):
    return features.TensorInfo(
        shape=(None,)*self._num_dims, dtype=self._dtype)

  def get_serialized_info(self):
    return dict(
        stripped=self._stripped_feature.get_serialized_info(),
        padding=self._padding_feature.get_serialized_info(),
    )

  def decode_example(self, tfexample_data):
    stripped = tfexample_data['stripped']
    padding = tfexample_data['padding']
    return tf.pad(stripped, padding, **self._pad_kwargs)

  def encode_example(self, example_data):
    if self._strip_fn is not None:
        example_data = self._strip_fn(example_data)
    if isinstance(example_data, dict):
      if len(example_data) != 2:
        raise ValueError(
            'example_data should have exactly 2 keys, "stripped" and "padding",'
            'got %s' % str(tuple(example_data.keys())))
      example_data = (example_data[k] for k in ('stripped', 'padding'))
    
    stripped, padding = example_data
    return dict(
        stripped=self._stripped_feature.encode_example(stripped),
        padding=self._padding_feature.encode_example(padding))