from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
import os
import json
import zipfile


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


class ShapenetCoreConfig(tfds.core.BuilderConfig):

    @abc.abstractproperty
    def synset_id(self):
        raise NotImplementedError

    @abc.abstractmethod
    def features(self):
        raise NotImplementedError

    @abc.abstractmethod
    def loader(self, archive):
        """
        Get a function that loads examples from the archive.

        The returned callable should be able to be used in a context block and
        have signature `(model_path, model_id) -> features`, where `features`
        is of the form of `self.features()`.

        See `ExampleLoader` for abstract interface with base implementations.
        """
        raise NotImplementedError

    def supervised_keys(self):
        return None


class ExampleLoader(object):
    def __init__(self, archive):
        self.archive = archive

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    @abc.abstractmethod
    def __call__(self, model_path, model_id):
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


def load_taxonomy(path):
    with tf.io.gfile.GFile(path, 'r') as fp:
        return json.load(fp)


def load_splits(path, synset_id):
    splits = ('train', 'test', 'val')
    model_ids = {k: [] for k in splits}
    with tf.io.gfile.GFile(path, "r") as fp:
        fp.readline()    # header
        for line in fp.readlines():
            line = line.rstrip()
            if line == '':
                pass
            record_id, synset_id_, sub_synset_id, model_id, split = \
                line.split(',')
            del record_id, sub_synset_id
            if synset_id_ == synset_id:
                model_ids[split].append(model_id)
    model_ids['validation'] = model_ids.pop('val')
    return model_ids


class ShapenetCore(tfds.core.GeneratorBasedBuilder):
    BUILDER_CONFIGS = []

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            features=self.builder_config.features(),
            citation=SHAPENET_CITATION,
            supervised_keys=self.builder_config.supervised_keys(),
            urls=[SHAPENET_URL],
        )

    def _split_generators(self, dl_manager):
        synset_id = self.builder_config.synset_id
        synset_url = DL_URL.format(synset_id=synset_id)
        split_path = dl_manager.download(SPLIT_URL)
        zip_path = dl_manager.download(synset_url)

        model_ids = load_splits(split_path, synset_id)
        # print([(k, len(v)) for k, v in model_ids.items()])
        # exit()
        splits = sorted(model_ids.keys())
        return [tfds.core.SplitGenerator(
            name=split, num_shards=len(model_ids[split]) // 1000 + 2,
            gen_kwargs=dict(zip_path=zip_path, model_ids=model_ids[split]))
                        for split in splits]

    def _generate_examples(self, zip_path, model_ids):
        config = self.builder_config
        synset_id = config.synset_id
        with tf.io.gfile.GFile(zip_path, "rb") as fp:
            archive = zipfile.ZipFile(fp)
            with config.loader(archive) as loader:
                for model_id in model_ids:
                    model_path = os.path.join(synset_id, model_id, 'model.obj')
                    example = loader(model_path, model_id)
                    if example is not None:
                        yield example
