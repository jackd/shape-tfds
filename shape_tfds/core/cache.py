from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import os
import numpy as np


class DataCache(collections.Mapping):
    def __init__(self, feature):
        self._feature = feature

    def _save_data(self, key, data):
        raise NotImplementedError

    def _load_data(self, key):
        raise NotImplementedError

    def __setitem__(self, key, value):
        example_data = self._feature.encode_example(value)
        self._save_data(key, example_data)

    def __getitem__(self, key):
        return self._feature.decode_example(self._load_data(key))


class SubdirDataCache(DataCache):
    def __init__(self, root_dir, feature):
        self._root_dir = root_dir
        super(SubdirDataCache, self).__init__(feature)

    def _subdir(self, *args):
        return os.path.join(self._root_dir, *args)

    def _save_data(self, key, data):
        for k, v in self._feature.flatten(data).items():
            np.save(self._subdir(key, k), v)

    def _load_data(self, key):
        flat = {}
        for k, v in self._feature.flatten(self._feature).items():
            flat[k] = v.decode_example(
                np.load(os.path.join(self._subdir, key, k)))
        return self._feature._nest(flat)

    def keys(self):
        return os.listdir(self._root_dir)

    def __contains__(self, key):
        return os.path.isdir(self._subdir(key))

    def __len__(self):
        return len(self.keys())

    def __iter__(self):
        return iter(self.keys())
