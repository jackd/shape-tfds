from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import six
import numpy as np
import tensorflow as tf

import tensorflow_datasets.public_api as tfds
from tensorflow_datasets.core import utils as core_utils
from shape_tfds.shape.shapenet.core.base import SHAPENET_URL
import h5py
import collections

_PARTNET2019_URL = "https://cs.stanford.edu/~kaichun/partnet/"

_CITATION = """
@article{mo2018partnet,
    title={{PartNet}: A Large-scale Benchmark for Fine-grained and Hierarchical Part-level {3D} Object Understanding},
    author={Mo, Kaichun and Zhu, Shilin and Chang, Angel and Yi, Li and Tripathi, Subarna and Guibas, Leonidas and Su, Hao},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2019}
}
"""

DESCRIPTION = """
The PartNet dataset provides fine grained part annotation of objects in
ShapeNetCore. We provide 573,585 part instances over 26,671 3D models covering
24 object categories."""

OBJECT_CATEGORIES = (
    'bag',
    'bed',
    'bottle',
    'bowl',
    'chair',
    'clock',
    'dishwasher',
    'display',
    'door',
    'earphone',
    'faucet',
    'hat',
    'keyboard',
    'knife',
    'lamp',
    'laptop',
    'microwave',
    'mug',
    'refrigerator',
    'scissors',
    'storage_furniture',
    'table',
    'trash_can',
    'vase',
)


def to_upper_camel_case(snake_case):
    return ''.join((w.capitalize() for w in snake_case.split('_')))


def class_path(object_category, level=1):
    camel = to_upper_camel_case(object_category)
    return os.path.join(os.path.dirname(__file__), 'after_merging_label_ids',
                        '{}-level-{}.txt'.format(camel, level))


def class_names(object_category, level=1):
    path = class_path(object_category, level=level)
    names = ['unknown']
    with tf.io.gfile.GFile(path, 'r') as fp:
        lines = (l.strip().split(' ')[1] for l in fp.readlines())
        names.extend(l for l in lines if l)
    return names


def label_index_map(top_level_names, other_names):
    if not isinstance(other_names, dict):
        other_names = {k: i for i, k in enumerate(other_names)}

    def get_index(base_name):
        name = base_name
        while name not in other_names:
            name = '/'.join(name.split('/')[:-1])
            if not name:
                return 0  # unknown
        return other_names[name]

    return np.array([get_index(n) for n in top_level_names], dtype=np.int64)


def levels(object_category):
    return tuple(i for i in range(1, 4)
                 if tf.io.gfile.exists(class_path(object_category, level=i)))


class PartnetConfig(tfds.core.BuilderConfig):

    def __init__(self, object_category):
        self.object_category = object_category
        super(PartnetConfig, self).__init__(name=object_category,
                                            version=core_utils.Version("0.1.0"),
                                            description=object_category)


class Partnet(tfds.core.GeneratorBasedBuilder):
    URLS = [SHAPENET_URL, _PARTNET2019_URL]

    BUILDER_CONFIGS = [PartnetConfig(c) for c in OBJECT_CATEGORIES]

    @core_utils.memoized_property
    def class_names(self):
        return {
            level: class_names(self.builder_config.object_category, level)
            for level in self.levels
        }

    @core_utils.memoized_property
    def num_classes(self):
        names = self.class_names
        return {k: len(v) for k, v in names.items()}

    @core_utils.memoized_property
    def label_index_maps(self):
        class_names = self.class_names
        top_names = {k: i for i, k in enumerate(class_names[self.levels[-1]])}
        return {
            k: label_index_map(top_names, class_names[k])
            for k in self.levels[:-1]
        }

    def __init__(self, level=1, config=None, **kwargs):
        if config is None:
            config = Partnet.BUILDER_CONFIGS[0]
        elif isinstance(config, six.string_types):
            for c in Partnet.BUILDER_CONFIGS:
                if c.name == config:
                    config = c
                    break
            else:
                raise ValueError(
                    "config string {} doesn't made name of any available "
                    "BUILDER_CONFIGS.")
        self.levels = levels(config.object_category)

        if level not in self.levels:
            raise ValueError(
                'Invalid level {} for category "{}". Must be in {}'.format(
                    level, config.object_category, self.levels))
        self.level = level
        super(Partnet, self).__init__(config=config, **kwargs)

    def _split_generators(self, dl_manager):
        # instance seg: http://download.cs.stanford.edu/orion/partnet_dataset/ins_seg_h5.zip
        # meshes, point clouds, visualizations: http://download.cs.stanford.edu/orion/partnet_dataset/data_v0.zip
        path = dl_manager.download(
            'http://download.cs.stanford.edu/orion/partnet_dataset/sem_seg_h5.zip'
        )

        label_index_maps = self.label_index_maps
        gens = [
            tfds.core.SplitGenerator(
                name=split,
                gen_kwargs=dict(
                    archive_fn=lambda: dl_manager.iter_archive(path),
                    split=split,
                    label_index_maps=label_index_maps))
            for split in ('train', 'test', 'validation')
        ]
        return gens

    def _label_key(self, level):
        return 'label-{}'.format(level)

    def _generate_examples(self, archive_fn, split, label_index_maps):
        archive = archive_fn()
        if split == 'validation':
            split = 'val'
        camel = to_upper_camel_case(self.builder_config.object_category)
        for path, fp in archive:
            split_path = path.split('/')
            if len(split_path) < 3 or not split_path[1].startswith(
                    camel) or not (split_path[2].endswith('.h5') and
                                   split_path[2].startswith(split)):
                continue
            positions, labels = self._load_examples(fp)
            for i, (p, l) in enumerate(zip(positions, labels)):
                key = '{}-{}'.format(path[:-3], i)
                data = dict(path=path, index=i, positions=p)
                for level in self.levels[:-1]:
                    data[self._label_key(level)] = label_index_maps[level][l]
                data[self._label_key(self.levels[-1])] = l
                yield key, data

    def _load_examples(self, fp):
        import io
        fp = io.BytesIO(fp.read())
        fp = h5py.File(fp, 'r')
        positions = fp['data']
        labels = fp['label_seg']
        num_examples = positions.shape[0]
        if labels.shape[0] != num_examples:
            raise RuntimeError('Inconsistent shapes of positions/labels')
        positions = positions[:].astype(np.float32)
        labels = labels[:].astype(np.int64)
        num_classes = self.num_classes[self.levels[-1]]
        if np.any(labels >= num_classes):
            raise RuntimeError(
                'Invalid label value. Expected all to be in [0, num_classes), '
                'but {} >= {}'.format(np.max(labels), num_classes))
        if np.any(labels < 0):
            raise RuntimeError(
                'Invalid label value. Expected all to be non-negative but got '
                '{}'.format(np.min(labels)))

        return positions, labels

    def _info(self):
        path = tfds.core.features.Text()
        index = tfds.core.features.Tensor(dtype=tf.int64, shape=())
        positions = tfds.core.features.Tensor(dtype=tf.float32,
                                              shape=(10000, 3))
        features = dict(path=path, index=index, positions=positions)
        labels = tfds.core.features.Tensor(shape=(10000,), dtype=tf.int64)
        for level in self.levels:
            features[self._label_key(level)] = labels
        features = tfds.features.FeaturesDict(features)

        return tfds.core.DatasetInfo(
            builder=self,
            description=DESCRIPTION,
            features=features,
            citation=_CITATION,
            supervised_keys=("positions", self._label_key(self.level)),
            urls=self.URLS)
