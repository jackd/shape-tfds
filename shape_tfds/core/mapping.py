from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import collections
import os
import numpy as np
import tensorflow as tf
import h5py

class DirectoryMapping(collections.Mapping):
    def __init__(self, root_dir):
        self._root_dir = root_dir

    def _path(self, key):
        return os.path.join(self._root_dir, '%s.npy' % key)

    def __getitem__(self, key):
        return np.load(self._path(key))

    def __setitem__(self, key, value):
        path = self._path(key)
        dirname = os.path.dirname(path)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        return np.save(path, value)

    def __contains__(self, key):
        return os.path.isfile(self._path(key))

    def keys(self):
        n = len(self._root_dir) + 1
        for dirname, _, fns in os.walk(self._root_dir):
            for fn in fns:
                if fn.endswith('.npy'):
                    path = os.path.join(dirname, fn)
                    if len(path) >= (n+4):
                        yield path[n:-4]

    def __iter__(self):
        return iter(self.keys())

    def __delitem__(self, key):
        os.remove(self._path(key))


class ZipMapping(collections.Mapping):
    def __init__(self, zipfile, root_dir=''):
        self._zipfile = zipfile
        self._root_dir = root_dir

    def _path(self, key):
        return os.path.join(self._root_dir, '%s.npy' % key)

    def __getitem__(self, key):
        with self._zipfile.open(self._path(key), 'r') as fp:
            return np.load(fp)

    def __setitem__(self, key, value):
        with self._zipfile.open(self._path(key), 'w') as fp:
            np.save(fp, value)

    def __contains__(self, key):
        try:
            with self._zipfile.open(self._path(key), 'r'):
                pass
            return True
        except KeyError:
            return False

    def __iter__(self):
        return iter(self.keys())

    def keys(self):
        return (n for n in self._zipfile.namelist() if n.endswith('.npy'))

    def len(self):
        return sum(1 for _ in self.keys())

    def __delitem__(self, key):
        raise RuntimeError('delete item not supported by `ZipMapping`s.')


def _h5_dataset_keys(group):
    for k, v in group.items():
        if isinstance(v, h5py.Dataset):
            yield k
        else:
            for rest in _h5_dataset_keys(v):
                yield os.path.join(k, rest)


class H5Mapping(collections.Mapping):
    def __init__(self, root):
        self._root = root

    def __getitem__(self, key):
        ds = self._root[key]
        if not isinstance(ds, h5py.Dataset):
            raise KeyError('Invalid key %s' % key)
        return np.array(ds)

    def __contains__(self, key):
        return isinstance(self._root.get(key, None), h5py.Dataset)

    def __setitem__(self, key, value):
        self._root.create_dataset(key, data=value)

    def keys(self):
        return _h5_dataset_keys(self._root)

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())

    def __delitem__(self, key):
        del self._root[key]


def flatten_dict(nested_dict):
    out = {}
    for k, v in nested_dict.items():
        if hasattr(v, 'items'):
            for k2, v2 in flatten_dict(v).items():
                out[os.path.join(k, k2)] = v2
        else:
            out[k] = v
    return out


def nest_dict(flat_dict):
    out = {}
    for k, v in flat_dict.items():
        split_k = k.split('/')
        if len(split_k) == 1:
            out[k] = v
        else:
            curr = out
            for subk in split_k[:-1]:
                curr = curr.setdefault(subk, {})
            curr[split_k[-1]] = v
    return out


class FeatureDecoder(object):
    def __init__(self, feature):
        self._feature = feature
        self._sess = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def open(self):
        if self._sess is not None:
            raise RuntimeError('Cannot open DecoderContext: already open')
        graph = tf.compat.v1.Graph()
        with graph.as_default():  # pylint: disable=not-context-manager
            with tf.device('/cpu:0'):
                self._components = {
                    k: tf.compat.v1.placeholder(
                        name=k, dtype=v.dtype, shape=v.shape)
                    for k, v in
                    flatten_dict(self._feature.get_serialized_info()).items()}
                self._decoded = self._feature.decode_example(
                    nest_dict(self._components))

        self._sess = tf.compat.v1.Session(graph=graph)

    def close(self):
        if self._sess is None:
            raise RuntimeError('Cannot close DecoderContext: already closed')
        self._sess.close()
        self._sess = None

    def __call__(self, serialized_values):
        feed_dict = {
            self._components[k]: v for k, v in serialized_values.items()}
        try:
            return self._sess.run(self._decoded, feed_dict=feed_dict)
        except Exception:
            np.save('/tmp/brle.npy', serialized_values['voxels/stripped/base'])
            raise RuntimeError(
                'Error computing decoding with values %s'
                % str(serialized_values))


class FeatureMapping(collections.Mapping):
    """
    Class for mapping Feature data to a component mapping.

    Example usage:
    ```python
    with h5py.File('feature_cache.h5', 'w') as group:
        component_mapping = H5Mapping(group)
        feature_mapping = FeatureMapping(feature, component_mapping)
        for k, v in data_to_store:
            feature_mapping[k] = v

    ######

    with h5py.File('feature_cache.h5', 'r') as group:
        component_mapping = H5Mapping(group)
        feature_mapping = FeatureMapping(feature, component_mapping)
        np_data = feature_mapping[some_key]
    ```
    """

    def __init__(self, feature, component_mapping):
        """
        Args:
            feature: tfds TopLevelFeature
            component_mapping: collections.Mapping instance for getting/setting
                stored numpy data.
        """
        self._component_mapping = component_mapping
        self._feature = feature
        self._serialized_info = feature.get_serialized_info()
        self._decoder = FeatureDecoder(feature)
        self._flat_info = flatten_dict(self._serialized_info)
        self._flat_keys = tuple(sorted(self._flat_info))

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def open(self):
        self._decoder.open()

    def close(self):
        self._decoder.close()

    def __getitem__(self, key):
        components = {k: self._component_mapping[os.path.join(key, k)]
                      for k in self._flat_keys}
        return self._decoder(components)

    def __setitem__(self, key, value):
        value = self._feature.encode_example(value)
        for k, v in flatten_dict(value).items():
            self._component_mapping[os.path.join(key, k)] = v

    def keys(self):
        return set(k.split('/')[0] for k in self._component_mapping.keys())

    def __iter__(self):
        return iter(self.keys())

    def __contains__(self, key):
        return os.path.join(key, self._flat_keys[0]) in self._component_mapping

    def __len__(self):
        return sum(1 for _ in self.keys())
