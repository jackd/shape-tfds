from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow_datasets.core import features
from tensorflow_datasets.core.features import top_level_feature
from shape_tfds.core import util


def strip_zeros(example_data):
    import trimesh
    stripped, padding = trimesh.voxel.encoding.DenseEncoding(  # pylint: disable=unpacking-non-sequence
        example_data).stripped
    return dict(stripped=stripped.dense, padding=padding)


class PaddedTensor(features.FeatureConnector):

    def __init__(self,
                 stripped_feature,
                 pad_kwargs=None,
                 strip_fn=strip_zeros,
                 padded_shape=None):
        self._stripped = stripped_feature
        stripped_info = stripped_feature.get_tensor_info()
        self._num_dims = len(stripped_info.shape)
        self._dtype = stripped_info.dtype
        self._padding = features.feature.Tensor(shape=(self._num_dims, 2),
                                                dtype=tf.int64)
        self._pad_kwargs = {} if pad_kwargs is None else pad_kwargs
        self._strip_fn = strip_fn
        if padded_shape is not None and len(padded_shape) != self._num_dims:
            raise ValueError(
                'padded_shape and stripped_feature shapes inconsistent.')
        self._padded_shape = padded_shape
        # self._set_top_level()

    def get_tensor_info(self):
        shape = ((None,) * self._num_dims
                 if self._padded_shape is None else self._padded_shape)
        return features.TensorInfo(shape=shape, dtype=self._dtype)

    def decode_example(self, tfexample_data):
        tfexample_data = util.nest_dicts(tfexample_data)
        stripped = self._stripped.decode_example(tfexample_data['stripped'])
        padding = self._padding.decode_example(tfexample_data['padding'])
        out = tf.pad(stripped, padding, **self._pad_kwargs)
        if self._padded_shape is not None:
            out.set_shape(self._padded_shape)
        return out

    def encode_example(self, example_data):
        if self._strip_fn is not None and isinstance(example_data, np.ndarray):
            example_data = self._strip_fn(example_data)
        return dict(
            stripped=self._stripped.encode_example(example_data['stripped']),
            padding=self._padding.encode_example(example_data['padding']))

    def get_serialized_info(self):
        return util.flatten_dicts(
            dict(stripped=self._stripped.get_serialized_info(),
                 padding=self._padding.get_serialized_info()))
