from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
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


@contextlib.contextmanager
def mesh_loader_context(synset_id, dl_manager=None, map_fn=None):
    """
    Get a mesh loading context.

    Delays downloading relevant zip files until opened.

    Example usage:
    ```python
    loader = mapped_mesh_loader_context(synset_id)
    with loader:
        # possible download starts
        for key in loader:
            print(key)
            scene = loader[key]  # trimesh.Scene
            scene.show()
    ```
    """
    zip_path = get_obj_zip_path(synset_id, dl_manager)
    try:
        fp = tf.io.gfile.GFile(zip_path, "rb")
        yield mesh_loader(zipfile.ZipFile(fp)).map(map_fn)
    finally:
        fp.close()


class ShapenetCoreConfig(tfds.core.BuilderConfig):
    def obj_path(self, model_id):
        """path of obj files in shapenet core zip files."""
        return os.path.join(self.synset_id, model_id, 'model.obj')

    @abc.abstractproperty
    def synset_id(self):
        raise NotImplementedError

    @abc.abstractmethod
    def features(self):
        """python dict of features. Should not contain model_id."""
        raise NotImplementedError

    def supervised_keys(self):
        return None

    @abc.abstractmethod
    def loader(self, dl_manager=None):
        """
        Get a loading context manager.

        Loaded example data should be encodable by `self.features()`.

        Example usage:
        ```python
        with config.loader(dl_manager) as loader:
            for model_id in model_ids:
                example_data = loader[model_id]
        ```

        Example implementation:
        ```python
        def loader(self, dl_manager):
            zip_path = get_obj_zip_path(
                synset_id=self.synset_id, dl_manager=dl_manager)
            return mesh_loader_context(zip_path, map_fn=custom_map_fn)
        ```

        Args:
            zip_path: zipfile with models at SYNSET_ID/model_id/model.obj
            model_id: id of the model.

        Returns:
            context manager with `__enter__` returning a mapping from
            `model_id -> example_data`, where `example_data` is a mapping and
            `example_data[k]` is encodable by `self.features()[k]`.
        """
        raise NotImplementedError


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


BASE_URL = 'http://shapenet.cs.stanford.edu/shapenet/obj-zip/'
DL_URL = '%s/ShapeNetCore.v1/{synset_id}.zip' % BASE_URL
SPLIT_URL = '%s/SHREC16/all.csv' % BASE_URL
TAXONOMY_URL = '%s/ShapeNetCore.v1/taxonomy.json' % BASE_URL


def _load_taxonomy(path):
    with tf.io.gfile.GFile(path, 'r') as fp:
        return json.load(fp)


def _load_splits_ids(path):
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
            del record_id, sub_synset_id
            split_dicts.setdefault(synset_id, {}).setdefault(split, []).append(
                model_id)

    for split_ids in split_dicts.values():
        split_ids['validation'] = split_ids.pop('val')
    return split_dicts


class ShapenetCore(tfds.core.GeneratorBasedBuilder):
    BUILDER_CONFIGS = []

    def _info(self):
        features = self.builder_config.features()
        features['model_id'] = tfds.core.features.Text()
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.core.features.FeaturesDict(features),
            citation=SHAPENET_CITATION,
            supervised_keys=self.builder_config.supervised_keys(),
            urls=[SHAPENET_URL],
        )

    def _split_generators(self, dl_manager):
        config = self.builder_config
        synset_id = config.synset_id
        model_ids = load_split_ids(dl_manager)[synset_id]
        splits = sorted(model_ids.keys())
        loader_fn = functools.partial(
            config.loader, dl_manager=dl_manager)

        return [tfds.core.SplitGenerator(
            name=split, num_shards=len(model_ids[split]) // 1000 + 2,
            gen_kwargs=dict(
                loader_fn=loader_fn, model_ids=model_ids[split]))
                for split in splits]

    def _generate_examples(self, **kwargs):
        gen = self._generate_example_data(**kwargs)
        return (
            ((v['model_id'], v) for v in gen)
            if self.version.implements(tfds.core.Experiment.S3) else gen)

    def _generate_example_data(self, loader_fn, model_ids):
        with loader_fn() as loader:
            for model_id in model_ids:
                example_data = loader[model_id]
                assert('model_id' not in example_data)
                example_data['model_id'] = model_id
                yield example_data


def _dl_manager():
    return tfds.core.download.DownloadManager(
        download_dir=os.path.join(tfds.core.constants.DATA_DIR, 'downloads'))


def load_split_ids(dl_manager=None):
    dl_manager = dl_manager or _dl_manager()
    return _load_splits_ids(dl_manager.download(SPLIT_URL))


def load_taxonomy(dl_manager=None):
    dl_manager = dl_manager or _dl_manager()
    return _load_taxonomy(dl_manager.download(TAXONOMY_URL))


def get_obj_zip_path(synset_id, dl_manager=None):
    """Get path of zip file containing obj files."""
    dl_manager = dl_manager or _dl_manager()
    return dl_manager.download(DL_URL.format(synset_id=synset_id))
