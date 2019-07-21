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
        self._flat_features = feature._flatten(feature)
        for k in self._flat_features:
            self._first_key = k
            break
        self._sess = None

    def __enter__(self):
        graph = tf.compat.v1.Graph()
        with graph.as_default():  # pylint: disable=not-context-manager
            with tf.device('/cpu:0'):
                components = {
                    k: tf.compat.v1.placeholder(
                        name=k, dtype=v.dtype, shape=v.shape)
                    for k, v in self._feature.items()}
                self._decoded = self._feature.decode_example(components)

        self._sess = tf.compat.v1.Session(graph=graph)
        return self

    def __exit__(self, *args, **kwargs):
        self._sess.close()
        self._sess = None

    def __getitem__(self, key):
        if self._sess is None:
            raise RuntimeError('FeatureMaping closed')
        components = self._feature._nest(
            {k: self._component_mapping[os.path.join(key, k)]
            for k in self._flat_features})
        return self._sess.run(self._decoded, feed_dict=components)

    def __setitem__(self, key, value):
        for k, v in self._feature._flatten(
                self._feature.encode_example(value)).items():
            self._component_mapping[os.path.join(key, k)] = v

    def keys(self):
        split_keys = (k.split('/') for k in self._component_mapping.keys())
        return (
            os.path.join(ks[:-1]) for ks in split_keys
            if ks[-1] == self._first_key)

    def __iter__(self):
        return iter(self.keys())

    def __contains__(self, key):
        return os.path.join(key, self._first_key) in self._component_mapping
