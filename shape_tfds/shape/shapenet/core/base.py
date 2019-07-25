from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from absl import logging
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


# class LengthedGenerator(object):
#     def __init__(self, generator, length):
#         self._generator = generator
#         self._length = length

#     def __len__(self):
#         return self._length

#     def __iter__(self):
#         return iter(self._generator)


def as_mesh(scene_or_mesh):
    import trimesh
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            tuple(scene_or_mesh.geometry.values()))
    else:
        mesh = scene_or_mesh
    return mesh


def mesh_loader(zipfile):
    from collection_utils.mapping import Mapping
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
        return trimesh.load(
            zipfile.open(obj), file_type='obj', resolver=resolver)

    return Mapping.mapped(keys, load_fn)


class MappedFileContext(object):
    def __init__(self, path, map_fn, on_open=None, on_close=None, mode='rb'):
        self._path = path
        self._map_fn = map_fn
        self._mode = mode
        self._fp = None
        self._args = None
        self._on_open = on_open
        self._on_close = on_close

    def __enter__(self):
        self._fp = tf.io.gfile.GFile(self._path, mode=self._mode)
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


def mesh_loader_context(synset_id, dl_manager=None, item_map_fn=None):
    """
    Get a mesh loading context.

    Delays downloading relevant zip files until opened.

    Example usage:
    ```python
    loader_context = mesh_loader_context(synset_id)
    with loader_context as loader:
        # possible download starts
        for key in loader:
            print(key)
            scene = loader[key]  # trimesh.Scene
            scene.show()
    ```
    """
    def file_map_fn(fp):
        loader = mesh_loader(zipfile.ZipFile(fp))
        if item_map_fn is not None:
            loader = loader.item_map(item_map_fn)
        return loader, None

    return MappedFileContext(
        get_obj_zip_path(synset_id, dl_manager), map_fn=file_map_fn)


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
    return synset_ids, synset_names


def load_bad_ids():
    import json
    path = os.path.join(os.path.dirname(__file__), 'bad_ids.json')
    with tf.io.gfile.GFile(path, "r") as fp:
        return json.load(fp)


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
    return tfds.core.download.DownloadManager(
        download_dir=os.path.join(tfds.core.constants.DATA_DIR, 'downloads'))


def load_split_ids(dl_manager=None, excluded='bad_ids'):
    dl_manager = dl_manager or _dl_manager()
    if excluded == 'bad_ids':
        excluded = load_bad_ids()
    return _load_splits_ids(dl_manager.download(SPLIT_URL), excluded)


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
    def loader_context(self, dl_manager=None):
        raise NotImplementedError


def get_data_mapping_context(
        config, data_dir=None, dl_manager=None, overwrite=False,
        item_map_fn=None):
    import h5py
    from shape_tfds.core import mapping
    import tqdm
    if data_dir is None:
        data_dir = os.path.join(
            tfds.core.constants.DATA_DIR, 'shapenet_core',
            'mappings', config.name, str(config.version))
    data_dir = os.path.expanduser(data_dir)
    if not tf.io.gfile.isdir(data_dir):
        tf.io.gfile.makedirs(data_dir)
    path = os.path.join(data_dir, 'mapping.h5')

    def file_map_fn(fp, mode='r'):
        h5 = h5py.File(fp, mode=mode)
        feature = tfds.core.features.FeaturesDict(config.features)
        feature._set_top_level()
        feature_mapping = mapping.FeatureMapping(
            feature, mapping.H5Mapping(h5))
        if item_map_fn is not None:
            from collection_utils.mapping import ItemMappedMapping
            return (
                ItemMappedMapping(feature_mapping, item_map_fn),
                feature_mapping)
        else:
            return feature_mapping, feature_mapping

    if not tf.io.gfile.exists(path) or overwrite:
        try:
            with h5py.File(path, 'w') as h5:
                feature = tfds.core.features.FeaturesDict(config.features)
                feature._set_top_level()
                dst = mapping.FeatureMapping(feature, mapping.H5Mapping(h5))
                with config.loader_context(dl_manager) as src:
                    for k, v in tqdm.tqdm(
                            src.items(),
                            total=len(src),
                            desc='Creating data mapping for %s' % config.name):
                        dst[k] = v
        except (Exception, KeyboardInterrupt):
            if tf.io.gfile.exists(path):
                tf.io.gfile.remove(path)
            raise

    return MappedFileContext(
        path, file_map_fn,
        on_open=lambda loader: loader.open(),
        on_close=lambda loader: loader.close())


class ShapenetCore(tfds.core.GeneratorBasedBuilder):
    def _info(self):
        features = self.builder_config.features
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
        loader_context = config.loader_context(dl_manager=dl_manager)

        gens = [tfds.core.SplitGenerator(
            name=split, num_shards=len(model_ids[split]) // 500 + 1,
            gen_kwargs=dict(
                loader_context=loader_context, model_ids=model_ids[split]))
                for split in splits]
        # we add num_examples for better progress bar info
        # this may cause issues if not every model generates an example
        for gen in gens:
            gen.split_info.statistics.num_examples = len(
                model_ids[gen.name])
        return gens

    def _generate_examples(self, model_ids, **kwargs):
        gen = self._generate_example_data(model_ids=model_ids, **kwargs)
        if (
                hasattr(self.version, 'implements') and
                self.version.implements(tfds.core.Experiment.S3)):
            gen = ((v['model_id'], v) for v in gen)
        # return LengthedGenerator(gen, len(model_ids))
        return gen

    def _generate_example_data(self, loader_context, model_ids):
        with loader_context as loader:
            for model_id in model_ids:
                try:
                    example_data = loader[model_id]
                except Exception:
                    logging.error(
                        'Error loading example %s / %s' %
                        (self.builder_config.synset_id, model_id))
                    raise
                if example_data is not None:
                    assert('model_id' not in example_data)
                    example_data['model_id'] = model_id
                    yield example_data
