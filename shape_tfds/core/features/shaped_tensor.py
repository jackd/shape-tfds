from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow_datasets.core.features import top_level_feature
from tensorflow_datasets.core import features
from shape_tfds.core import util


def _assert_reshapable(src_shape, dst_shape):
    src_n, dst_n = (np.prod([s
                             for s in shape
                             if s is not None])
                    for shape in (src_shape, dst_shape))
    if any(s is not None for s in src_shape) and src_n % dst_n is not None:
        raise ValueError(
            'Number of elements inconsistent for reshaping, got %s and %s' %
            (str(src_shape), str(dst_shape)))


class StaticShapedTensor(top_level_feature.TopLevelFeature):

    def __init__(self, base_feature, shape):
        if not isinstance(shape, tuple):
            raise ValueError('shape must be a tuple, got %s' % str(shape))
        num_unknown = shape.count(None)
        self._base_info = base_feature.get_tensor_info()
        flat_shape = self._base_info.shape
        if num_unknown == 0:
            if None in flat_shape:
                raise ValueError(
                    'Cannot assign fully known static base_feature with partially '
                    'unknown shape')
        elif num_unknown == 1:
            _assert_reshapable(flat_shape, shape)
        else:
            raise ValueError(
                'shape must contain at most 1 None. Use DynamicShapedTensor for more '
                'unknown dimensions.')
        self._base_feature = base_feature
        self._shape = shape

    def get_tensor_info(self):
        return features.TensorInfo(shape=self._shape,
                                   dtype=self._base_info.dtype)

    def encode_example(self, example_data):
        return self._base_feature.encode_example(
            np.reshape(example_data,
                       [-1 if s is None else s for s in self._base_info.shape]))

    def decode_example(self, tfexample_data):
        base = self._base_feature.decode_example(tfexample_data)
        return tf.reshape(base, [-1 if s is None else s for s in self._shape])

    def get_serialized_info(self):
        """See base class for details."""
        return self._base_feature.get_serialized_info()

    def save_metadata(self, data_dir, feature_name):
        return self._base_feature.save_metadata(data_dir,
                                                '%s-base' % feature_name)

    def load_metadata(self, data_dir, feature_name):
        return self._base_feature.load_metadata(data_dir,
                                                '%s-base' % feature_name)


class DynamicShapedTensor(features.FeatureConnector):

    def __init__(self, base_feature, shape):
        if not isinstance(shape, tuple):
            raise ValueError('shape must be a tuple, got %s' % str(shape))
        num_unknown = shape.count(None)
        self._base_info = base_feature.get_tensor_info()
        if num_unknown < 2:
            raise ValueError(
                'shape must contain at least 2 None values, got %s. Use '
                'StaticShapedTensor.' % shape)
        base_shape = self._base_info.shape
        _assert_reshapable(base_shape, shape)
        self._shape_tuple = shape
        self._base = base_feature
        self._shape = features.Tensor(shape=(len(shape),), dtype=tf.int64)

    def get_tensor_info(self):
        return features.TensorInfo(shape=self._shape_tuple,
                                   dtype=self._base_info.dtype)

    def get_serialized_info(self):
        return util.flatten_dicts(
            dict(base=self._base.get_serialized_info(),
                 shape=self._shape.get_serialized_info()))

    def decode_example(self, tfexample_data):
        tfexample_data = util.nest_dicts(tfexample_data)
        return tf.reshape(self._base.decode_example(tfexample_data['base']),
                          self._shape.decode_example(tfexample_data['shape']))

    def encode_example(self, example_data):
        base_data = np.reshape(
            example_data,
            [-1 if s is None else s for s in self._base_info.shape])
        return util.flatten_dicts(
            dict(base=self._base.encode_example(base_data),
                 shape=self._shape.encode_example(example_data.shape)))


def ShapedTensor(base_feature, shape):
    return (StaticShapedTensor if shape.count(None) < 2 else
            DynamicShapedTensor)(base_feature, shape)
