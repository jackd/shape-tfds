from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from shape_tfds.shape.modelnet import base


class Pointnet2Config(base.CloudNormalConfig):

    def __init__(self, num_classes):
        """num_classes must be 10 or 40."""
        assert (num_classes in (10, 40))
        num_points = 10000
        self._num_classes = num_classes
        super(Pointnet2Config, self).__init__(
            name="presampled%d-%d" % (num_classes, num_points),
            num_points=num_points,
            description=(
                "%d-class sampled 10000-point cloud used by PointNet++" %
                num_classes),
            version=tfds.core.utils.Version("0.0.1"))

    @property
    def num_classes(self):
        return self._num_classes

    def load_example(self, path):
        raise NotImplementedError


_BUILDER_CONFIGS = {n: Pointnet2Config(num_classes=n) for n in [10, 40]}


def get_config(num_classes=40):
    return _BUILDER_CONFIGS[num_classes]


class Pointnet2(tfds.core.GeneratorBasedBuilder):
    URLS = [base._URL_BASE, "http://stanford.edu/~rqi/pointnet2/"]
    BUILDER_CONFIGS = [_BUILDER_CONFIGS[n] for n in (10, 40)]
    _CITATION = """\
@article{qi2017pointnetplusplus,
    title={PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
    author={Qi, Charles R and Yi, Li and Su, Hao and Guibas, Leonidas J},
    journal={arXiv preprint arXiv:1706.02413},
    year={2017}
}
"""

    @property
    def up_dim(self):
        return 1

    @property
    def num_classes(self):
        return self.builder_config.num_classes

    def _info(self):
        example_index = tfds.features.Tensor(shape=(), dtype=tf.int64)
        class_names_path = base.get_class_names_path(self.num_classes)
        label = tfds.features.ClassLabel(names_file=class_names_path)
        features = {
            "label": label,
            "example_index": example_index,
        }
        inp_key, inp_feature = self.builder_config.feature_item
        features[inp_key] = inp_feature

        supervised_keys = (inp_key, "label")

        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.core.features.FeaturesDict(features),
            citation=self._CITATION,
            supervised_keys=supervised_keys,
            urls=self.URLS,
        )

    def _split_generators(self, dl_manager):
        res = "https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip"
        data_dir = dl_manager.download_and_extract(res)
        data_dir = os.path.join(data_dir, "modelnet40_normal_resampled")
        out = []
        num_classes = self.num_classes
        for split, key, num_shards in (
            (tfds.Split.TRAIN, "train", 4 * num_classes // 10),
            (tfds.Split.TEST, "test", 1 * num_classes // 10),
        ):
            split_path = os.path.join(data_dir,
                                      "modelnet%d_%s.txt" % (num_classes, key))
            with tf.io.gfile.GFile(split_path, "r") as fp:
                example_ids = [l for l in fp.read().split("\n") if l != ""]
            gen = tfds.core.SplitGenerator(name=split,
                                           num_shards=num_shards,
                                           gen_kwargs=dict(
                                               example_ids=example_ids,
                                               data_dir=data_dir))
            gen.split_info.statistics.num_examples = len(example_ids)
            out.append(gen)
        return out

    def _generate_examples(self, example_ids, data_dir):
        for example_id in example_ids:
            split_id = example_id.split("_")
            label = "_".join(split_id[:-1])
            example_index = int(split_id[-1]) - 1
            path = os.path.join(data_dir, label, "%s.txt" % example_id)
            with tf.io.gfile.GFile(path, "rb") as fp:
                data = np.loadtxt(fp, delimiter=",", dtype=np.float32)
            positions, normals = np.split(data, 2, axis=1)  # pylint: disable=unbalanced-tuple-unpacking
            cloud = dict(positions=positions, normals=normals)
            yield ((label, example_index),
                   dict(cloud=cloud, label=label, example_index=example_index))
