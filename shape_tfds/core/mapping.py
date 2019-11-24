from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from absl import logging
import collections
import os
import numpy as np
import six
import tensorflow as tf
import tensorflow_datasets as tfds
import h5py
import contextlib
import functools
import tqdm
from PIL import Image


class ImmutableMapping(collections.Mapping):

    def __init__(self, base_mapping):
        self._base = base_mapping

    def __getitem__(self, key):
        return self._base[key]

    def __iter__(self):
        return iter(self._base)

    def __len__(self):
        return len(self._base)

    def __contains__(self, key):
        return key in self._base


class ShallowDirectoryMapping(collections.Mapping):

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
            dirname = str(dirname)
            for fn in fns:
                fn = str(fn)
                if fn.endswith('.npy'):
                    path = os.path.join(dirname, fn)
                    if len(path) >= (n + 4):
                        yield path[n:-4]

    def __iter__(self):
        return iter(self.keys())

    def __delitem__(self, key):
        os.remove(self._path(key))


def _is_file(path):
    return tf.io.gfile.exists(path) and not tf.io.gfile.isdir(path)


class DeepDirectoryMapping(collections.Mapping):

    def __init__(self, root_dir):
        self._root_dir = root_dir

    def _path(self, key):
        return os.path.join(self._root_dir, key)

    def __getitem__(self, key):
        path = self._path(key)
        if _is_file(path):
            return path
        else:
            raise KeyError(key)

    def __contains__(self, key):
        return _is_file(self._path(key))

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return sum(len(walk[2]) for walk in tf.io.gfile.walk(self._root_dir))

    def keys(self):
        n = len(self._root_dir) + 1
        for dirname, _, fns in os.walk(self._root_dir):
            dirname = str(dirname)
            for fn in fns:
                yield os.path.join(dirname[n:], str(fn))


def _image_to_string(image, img_format='png'):
    import trimesh
    with trimesh.util.BytesIO() as buffer:
        image.save(buffer, img_format)
        return buffer.getvalue()


class ImageDirectoryMapping(collections.Mapping):

    def __init__(self, root_dir, extension='png'):
        self._root_dir = root_dir
        self._extension = extension

    def keys(self):
        return (k for k in tf.io.gfile.listdir(self._root_dir)
                if k.endswith('.png'))

    def __len__(self):
        return len(tuple(self.keys()))

    def __iter__(self):
        return iter(self.keys())

    def __contains__(self, key):
        return tf.io.gfile.exists(self._path(key))

    def _path(self, key):
        return os.path.join(self._root_dir, '%s.png' % key)

    def __getitem__(self, key):
        path = self._path(key)
        if not tf.io.gfile.exists(path):
            raise KeyError('No file at path %s' % path)
        return path

    def __setitem__(self, key, value):
        if not tf.io.gfile.isdir(self._root_dir):
            tf.io.gfile.makedirs(self._root_dir)
        dst = self._path(key)
        if hasattr(value, 'read'):
            value = value.read()
        elif isinstance(value, Image.Image):
            value = _image_to_string(value)
        if isinstance(value, bytes):
            with tf.io.gfile.GFile(dst, 'wb') as fp:
                fp.write(value)
        elif isinstance(value, six.string_types):
            with tf.io.gfile.GFile(value, 'rb') as src:
                with tf.io.gfile.GFile(dst, 'wb') as fp:
                    fp.write(src.read())
        else:
            raise TypeError(value)

    def __delitem__(self, key):
        tf.io.gfile.remove(self._path(key))


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
                    k: tf.compat.v1.placeholder(name=k,
                                                dtype=v.dtype,
                                                shape=v.shape) for k, v in
                    flatten_dict(self._feature.get_serialized_info()).items()
                }
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
            self._components[k]: v for k, v in serialized_values.items()
        }
        try:
            return self._sess.run(self._decoded, feed_dict=feed_dict)
        except Exception:
            np.save('/tmp/brle.npy', serialized_values['voxels/stripped/base'])
            raise RuntimeError('Error computing decoding with values %s' %
                               str(serialized_values))


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
        components = {
            k: self._component_mapping[os.path.join(key, k)]
            for k in self._flat_keys
        }
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


class MappingConfig(tfds.core.BuilderConfig):

    @property
    def supervised_keys(self):
        return None

    @abc.abstractproperty
    def features(self):
        """dict of features (not including 'model_id')."""
        raise NotImplementedError

    @abc.abstractmethod
    def lazy_mapping(self, dl_manager=None):
        raise NotImplementedError

    @contextlib.contextmanager
    def cache_mapping(self, cache_dir, mode='r'):
        import h5py
        from shape_tfds.core import mapping
        feature = tfds.core.features.FeaturesDict(self.features)
        feature._set_top_level()
        path = os.path.join(cache_dir, 'cache.h5')
        if not tf.io.gfile.isdir(cache_dir):
            tf.io.gfile.makedirs(cache_dir)
        with h5py.File(path, mode=mode) as h5:
            with mapping.FeatureMapping(feature, mapping.H5Mapping(h5)) as fm:
                yield fm

    def create_cache(self, cache_dir, dl_manager=None, overwrite=False):
        with self.cache_mapping(cache_dir, mode='a') as cache:
            if not overwrite:
                keys = tuple(k for k in keys if k not in cache)
            if len(keys) == 0:
                return
            with self.lazy_mapping(dl_manager) as src:
                for k in tqdm.tqdm(keys, desc='creating cache'):
                    cache[k] = src[k]


def concat_dict_values(dictionary):
    return np.concatenate([dictionary[k] for k in sorted(dictionary)])


class MappingBuilder(tfds.core.GeneratorBasedBuilder):

    def __init__(self, from_cache=False, overwrite_cache=False, **kwargs):
        self._from_cache = from_cache
        self._overwrite_cache = overwrite_cache
        super(MappingBuilder, self).__init__(**kwargs)

    @abc.abstractproperty
    def key(self):
        raise NotImplementedError

    @abc.abstractproperty
    def key_feature(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _load_split_keys(self, dl_manager):
        raise NotImplementedError

    @abc.abstractproperty
    def citation(self):
        raise NotImplementedError

    @abc.abstractproperty
    def urls(self):
        raise NotImplementedError

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            features=self.features,
            citation=self.citation,
            supervised_keys=self.builder_config.supervised_keys,
            urls=self.urls,
        )

    @property
    def features(self):
        """features used in self._info."""
        key = self.key
        config = self.builder_config
        features = config.features
        assert (key not in features)
        features[key] = self.key_feature
        return tfds.core.features.FeaturesDict(features)

    @property
    def cache_dir(self):
        head, tail = os.path.split(self.data_dir)
        if 'incomplete' in tail:
            tail = '.'.join(tail.split('.')[:-1])
        return os.path.join(head, 'cache', tail)

    def create_cache(self, dl_manager=None):
        self.builder_config.create_cache(cache_dir=self.cache_dir,
                                         dl_manager=dl_manager,
                                         overwrite=self._overwrite_cache)

    def remove_cache(self):
        tf.io.gfile.rmtree(self.cache_dir)

    def _split_generators(self, dl_manager):
        config = self.builder_config
        keys = self._load_split_keys(dl_manager)
        splits = sorted(keys.keys())

        if self._from_cache:
            self.create_cache(dl_manager=dl_manager)
            mapping_fn = functools.partial(config.cache_mapping,
                                           cache_dir=self.cache_dir,
                                           mode='r')
        else:
            mapping_fn = functools.partial(config.lazy_mapping,
                                           dl_manager=dl_manager)

        gens = [
            tfds.core.SplitGenerator(name=split,
                                     num_shards=len(keys[split]) // 500 + 1,
                                     gen_kwargs=dict(mapping_fn=mapping_fn,
                                                     keys=keys[split]))
            for split in splits
        ]
        # we add num_examples for better progress bar info
        for gen in gens:
            gen.split_info.statistics.num_examples = len(keys[gen.name])
        return gens

    def _generate_examples(self, keys, **kwargs):
        # wraps _generate_example_data, adding key if
        # `self.version.implements(tfds.core.Experiment.S3)`
        gen = self._generate_example_data(keys=keys, **kwargs)
        if (hasattr(self.version, 'implements') and
                self.version.implements(tfds.core.Experiment.S3)):
            gen = ((v[self.key], v) for v in gen)
        return gen

    def _generate_example_data(self, mapping_fn, keys):
        key_str = self.key
        with mapping_fn() as mapping:
            for key in keys:
                try:
                    out = mapping[key]
                    assert (key not in out)
                    out[key_str] = key
                    yield out
                except Exception:
                    logging.error('Error loading example %s' % key)
                    raise
