from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from absl import logging
import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
import tqdm
import os
import json
import zipfile
from shape_tfds.core import util
import six
import collections
import itertools
import contextlib
import functools

from collection_utils.mapping import Mapping
from collection_utils.iterable import single

DOWNLOADS_DIR = os.path.join(tfds.core.constants.DATA_DIR, 'downloads')

_bad_ids = {
    '04090263': (
        '4a32519f44dc84aabafe26e2eb69ebf4',  # empty
    ),
    # '02958343': (
    #     'e23ae6404dae972093c80fb5c792f223',  # too big
    # )  # resolved in fix/objvert
}


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


id_sets = ImmutableMapping({'bad_ids': ImmutableMapping(_bad_ids)})


SHAPENET_CITATION = """\
@article{chang2015shapenet,
    title={Shapenet: An information-rich 3d model repository},
    author={Chang, Angel X and Funkhouser, Thomas and Guibas, Leonidas and
            Hanrahan, Pat and Huang, Qixing and Li, Zimo and
            Savarese, Silvio and Savva, Manolis and Song, Shuran and
            Su, Hao and others},
    journal={arXiv preprint arXiv:1512.03012},
    year={2015}
}
"""

SHAPENET_URL = "https://www.shapenet.org/"


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, all texture information is lost.
    """
    import trimesh
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        mesh = scene_or_mesh
    return mesh


def zipped_mesh_loader(zipfile):
    namelist = zipfile.namelist()
    if len(namelist) == 0:
        raise ValueError('No entries in namelist')
    synset_id = namelist[0].split('/')[0]
    keys = set(
        n.split('/')[1] for n in namelist if n.endswith('.obj'))

    def load_fn(key):
        from shape_tfds.core.resolver import ZipSubdirResolver
        import trimesh
        subdir = os.path.join(synset_id, key)
        resolver = ZipSubdirResolver(zipfile, subdir)
        obj = os.path.join(subdir, 'model.obj')
        with zipfile.open(obj) as fp:
            return trimesh.load(fp, file_type='obj', resolver=resolver)

    return Mapping.mapped(keys, load_fn)


class Openable(object):
    """
    Base class for classes which can be used as context managers.

    Implement `_open` and `_close`.
    """
    def __init__(self):
        self._is_open = False

    def _open(self):
        pass

    def _close(self):
        pass

    @property
    def is_open(self):
        return self._is_open

    def open(self):
        if self.is_open:
            raise ValueError('Already open')
        self._open()
        self._is_open = True

    def close(self):
        if not self.is_open:
            raise ValueError('Already closed')
        self._close()
        self._is_open = False

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()


class MappedFileContext(object):
    def __init__(self, path, map_fn, on_open=None, on_close=None):
        self._path = path
        self._map_fn = map_fn
        self._fp = None
        self._args = None
        self._on_open = on_open
        self._on_close = on_close

    def __enter__(self):
        self._fp = tf.io.gfile.GFile(self._path, mode='rb')
        out, self._arg = self._map_fn(self._fp)
        if self._on_open is not None:
            self._on_open(self._arg)
        return out

    def __exit__(self, *args, **kwargs):
        if self._on_close is not None:
            self._on_close(self._arg)
        self._fp.close()
        self._mapped = None
        self._fp = None


def zipped_mesh_loader_context(synset_id, dl_manager=None, item_map_fn=None):
    """
    Get a mesh loading context.

    Delays downloading relevant zip files until opened.

    Does not extract files.

    Example usage:
    ```python
    loader_context = zipped_mesh_loader_context(synset_id)
    with loader_context as loader:
        # possible download starts
        for key in loader:
            print(key)
            scene = loader[key]  # trimesh.Scene
            scene.show()
    ```
    """
    def file_map_fn(fp):
        loader = zipped_mesh_loader(zipfile.ZipFile(fp))
        if item_map_fn is not None:
            loader = loader.item_map(item_map_fn)
        return loader, None

    return MappedFileContext(
        get_obj_zip_path(synset_id, dl_manager), map_fn=file_map_fn)


def extracted_mesh_paths(synset_id, dl_manager=None):
    dl_manager = dl_manager or _dl_manager()
    zip_path = get_obj_zip_path(synset_id, dl_manager)
    root_dir = dl_manager.extract(zip_path)
    synset_dir = os.path.join(root_dir, synset_id)
    assert(tf.io.gfile.isdir(synset_dir))
    model_ids = tf.io.gfile.listdir(synset_dir)
    return Mapping.mapped(
        tuple(model_ids),
        lambda key: os.path.join(synset_dir, key, 'model.obj'))


def load_synset_ids():
    path = os.path.join(os.path.dirname(__file__), 'core_synset.txt')
    synset_ids = {}
    synset_names = {}
    with tf.io.gfile.GFile(path, "rb") as fp:
        for line in fp.readlines():
            if hasattr(line, 'decode'):
                line = line.decode('utf-8')
            line = line.rstrip()
            if line == '':
                continue
            id_, names = line.split('\t')
            names = tuple(names.split(','))
            synset_names[id_] = names
            for n in names:
                synset_ids[n] = id_
    # repeated synset ids
    ambiguous_ids = {'bench': '02828884'}
    for k, v in ambiguous_ids.items():
        assert(k in synset_names[v])
        synset_ids[k] = v
    return synset_ids, synset_names


BASE_URL = 'http://shapenet.cs.stanford.edu/shapenet/obj-zip'
DL_URL = '%s/ShapeNetCore.v1/{synset_id}.zip' % BASE_URL
SPLIT_URL = '%s/SHREC16/all.csv' % BASE_URL
TAXONOMY_URL = '%s/ShapeNetCore.v1/taxonomy.json' % BASE_URL


def _load_taxonomy(path):
    with tf.io.gfile.GFile(path, 'r') as fp:
        return json.load(fp)


def _load_splits_ids(path, excluded):
    """Get a `dict: synset_id -> (dict: split -> model_ids)`."""
    split_dicts = {}
    with tf.io.gfile.GFile(path, "r") as fp:
        fp.readline()  # header
        for line in fp.readlines():
            line = line.rstrip()
            if line == '':
                pass
            record_id, synset_id, sub_synset_id, model_id, split = \
                line.split(',')
            if excluded and model_id in excluded.get(synset_id, ()):
                continue
            del record_id, sub_synset_id
            split_dicts.setdefault(synset_id, {}).setdefault(split, []).append(
                model_id)

    for split_ids in split_dicts.values():
        if 'val' in split_ids:
            if 'validation' in split_ids:
                raise RuntimeError('both "val" and "validation" keys found')
            else:
                split_ids['validation'] = split_ids.pop('val', None)

    return split_dicts


def _dl_manager():
    return tfds.core.download.DownloadManager(download_dir=DOWNLOADS_DIR)


def load_split_ids(dl_manager=None, excluded='bad_ids'):
    dl_manager = dl_manager or _dl_manager()

    return _load_splits_ids(
        dl_manager.download(SPLIT_URL),
        None if excluded is None else id_sets[excluded])


def load_taxonomy(dl_manager=None):
    dl_manager = dl_manager or _dl_manager()
    return _load_taxonomy(dl_manager.download(TAXONOMY_URL))


def get_obj_zip_path(synset_id, dl_manager=None):
    """Get path of zip file containing obj files."""
    dl_manager = dl_manager or _dl_manager()
    return dl_manager.download(DL_URL.format(synset_id=synset_id))


class ShapenetCoreConfig(tfds.core.BuilderConfig):
    def __init__(self, synset_id, **kwargs):
        self._synset_id = synset_id
        super(ShapenetCoreConfig, self).__init__(**kwargs)

    @property
    def synset_id(self):
        return self._synset_id

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

    def create_cache(
            self, cache_dir, model_ids=None, dl_manager=None, overwrite=False):
        if model_ids is None:
            model_ids = load_split_ids(dl_manager)[self.synset_id]
            model_ids = np.concatenate(
                [model_ids[k] for k in ('test', 'train', 'validation')])
        with self.cache_mapping(cache_dir, mode='a') as cache:
            if not overwrite:
                model_ids = tuple(k for k in model_ids if k not in cache)
            if len(model_ids) == 0:
                return
            with self.lazy_mapping(dl_manager) as src:
                for k in tqdm.tqdm(model_ids, desc='creating cache'):
                    cache[k] = src[k]


class ShapenetCore(tfds.core.GeneratorBasedBuilder):
    def __init__(self, from_cache=False, overwrite_cache=False, **kwargs):
        self._from_cache = from_cache
        self._overwrite_cache = overwrite_cache
        super(ShapenetCore, self).__init__(**kwargs)

    @property
    def cache_dir(self):
        head, tail = os.path.split(self.data_dir)
        if 'incomplete' in tail:
            tail = '.'.join(tail.split('.')[:-1])
        return os.path.join(head, 'cache', tail)

    def create_cache(self, model_ids=None, dl_manager=None):
        self.builder_config.create_cache(
            cache_dir=self.cache_dir,
            model_ids=model_ids,
            dl_manager=dl_manager,
            overwrite=self._overwrite_cache)

    def remove_cache(self):
        tf.io.gfile.rmtree(self.cache_dir)

    def _info(self):
        features = self.builder_config.features
        assert('model_id' not in features)
        features['model_id'] = tfds.core.features.Text()
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.core.features.FeaturesDict(features),
            citation=SHAPENET_CITATION,
            supervised_keys=self.builder_config.supervised_keys,
            urls=[SHAPENET_URL],
        )

    def _split_generators(self, dl_manager):
        config = self.builder_config
        synset_id = config.synset_id
        model_ids = load_split_ids(dl_manager)
        if synset_id in model_ids:
            model_ids = model_ids[synset_id]
        else:
            raise NotImplementedError
        splits = sorted(model_ids.keys())

        if self._from_cache:
            self.create_cache(
                dl_manager=dl_manager,
                model_ids=np.concatenate([model_ids[s] for s in splits]))
            mapping_fn = functools.partial(
                config.cache_mapping, cache_dir=self.cache_dir, mode='r')
        else:
            mapping_fn = functools.partial(
                config.lazy_mapping, dl_manager=dl_manager)

        gens = [tfds.core.SplitGenerator(
            name=split, num_shards=len(model_ids[split]) // 500 + 1,
            gen_kwargs=dict(mapping_fn=mapping_fn, model_ids=model_ids[split]))
                for split in splits]
        # we add num_examples for better progress bar info
        for gen in gens:
            gen.split_info.statistics.num_examples = len(model_ids[gen.name])
        return gens

    def _generate_examples(self, model_ids, **kwargs):
        # wraps _generate_example_data, adding model_id as a key if
        # `self.version.implements(tfds.core.Experiment.S3)`
        gen = self._generate_example_data(model_ids=model_ids, **kwargs)
        if (
                hasattr(self.version, 'implements') and
                self.version.implements(tfds.core.Experiment.S3)):
            gen = ((v['model_id'], v) for v in gen)
        return gen

    def _generate_example_data(self, mapping_fn, model_ids):
        with mapping_fn() as mapping:
            for model_id in model_ids:
                try:
                    out = mapping[model_id]
                    assert('model_id' not in out)
                    out['model_id'] = model_id
                    yield out
                except Exception:
                    logging.error(
                        'Error loading example %s / %s' %
                        (self.builder_config.synset_id, model_id))
                    raise
