import json
import os

import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from shape_tfds.shape.modelnet import base
from shape_tfds.shape.modelnet.base import _URL_BASE


class Pointnet2Config(base.CloudNormalConfig):
    def __init__(self, num_classes):
        """num_classes must be 10 or 40."""
        assert num_classes in (10, 40)
        num_points = 10000
        self._num_classes = num_classes
        super(Pointnet2Config, self).__init__(
            name="presampled%d-%d" % (num_classes, num_points),
            num_points=num_points,
            description=(
                "%d-class sampled 10000-point cloud used by PointNet++" % num_classes
            ),
            version=tfds.core.utils.Version("0.0.1"),
        )

    @property
    def num_classes(self):
        return self._num_classes

    def load_example(self, path):
        raise NotImplementedError


CONFIG10 = Pointnet2Config(num_classes=10)
CONFIG40 = Pointnet2Config(num_classes=40)

_BUILDER_CONFIGS = {
    10: CONFIG10,
    40: CONFIG40,
}


def get_config(num_classes=40):
    return _BUILDER_CONFIGS[num_classes]


class Pointnet2(tfds.core.GeneratorBasedBuilder):
    URLS = [_URL_BASE, "http://stanford.edu/~rqi/pointnet2/"]
    BUILDER_CONFIGS = [CONFIG10, CONFIG40]
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
            # urls=self.URLS,
            homepage="http://stanford.edu/~rqi/pointnet2/",
        )

    def _split_generators(self, dl_manager):
        res = "https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip"
        # data_dir = dl_manager.download_and_extract(res)
        data_dir = dl_manager.download_and_extract(res)
        data_dir = os.path.join(data_dir, "modelnet40_normal_resampled")
        out = []
        num_classes = self.num_classes
        for split, key in (
            (tfds.Split.TRAIN, "train"),
            (tfds.Split.TEST, "test"),
        ):
            split_path = os.path.join(
                data_dir, "modelnet{}_{}.txt".format(num_classes, key)
            )
            with tf.io.gfile.GFile(split_path, "r") as fp:
                example_ids = [l for l in fp.read().split("\n") if l != ""]
            gen = tfds.core.SplitGenerator(
                name=split, gen_kwargs=dict(example_ids=example_ids, data_dir=data_dir)
            )
            gen.split_info.statistics.num_examples = len(example_ids)
            out.append(gen)
        return out

    def _generate_examples(self, example_ids, data_dir):
        for example_id in example_ids:
            split_id = example_id.split("_")
            label = "_".join(split_id[:-1])
            example_index = int(split_id[-1]) - 1
            path = os.path.join(data_dir, label, "{}.txt".format(example_id))
            with tf.io.gfile.GFile(path, "rb") as fp:
                data = np.loadtxt(fp, delimiter=",", dtype=np.float32)
            positions, normals = np.split(  # pylint: disable=unbalanced-tuple-unpacking
                data, 2, axis=1
            )
            cloud = dict(positions=positions, normals=normals)
            yield (
                "{}-{}".format(label, example_index),
                dict(cloud=cloud, label=label, example_index=example_index),
            )


class Pointnet2H5(tfds.core.GeneratorBasedBuilder):
    """H5 variant - smaller, faster download/processing."""

    URLS = [_URL_BASE, "http://stanford.edu/~rqi/pointnet2/"]
    _CITATION = """\
@article{qi2017pointnetplusplus,
    title={PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
    author={Qi, Charles R and Yi, Li and Su, Hao and Guibas, Leonidas J},
    journal={arXiv preprint arXiv:1706.02413},
    year={2017}
}
"""
    VERSION = tfds.core.Version("0.0.1")

    @property
    def up_dim(self):
        return 1

    @property
    def num_classes(self):
        return 40

    @property
    def num_points(self):
        return 2048

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.core.features.FeaturesDict(
                {
                    "label": tfds.features.ClassLabel(
                        names_file=base.get_class_names_path(self.num_classes)
                    ),
                    "example_index": tfds.features.Tensor(shape=(), dtype=tf.int64),
                    "cloud": {
                        "positions": tfds.features.Tensor(
                            shape=(self.num_points, 3), dtype=tf.float32
                        ),
                        "normals": tfds.features.Tensor(
                            shape=(self.num_points, 3), dtype=tf.float32
                        ),
                        "face_indices": tfds.features.Tensor(
                            shape=(self.num_points,), dtype=tf.int64
                        ),
                    },
                }
            ),
            citation=self._CITATION,
            supervised_keys=("cloud", "label"),
            homepage="http://stanford.edu/~rqi/pointnet2/",
        )

    def _split_generators(self, dl_manager):
        res = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"
        data_dir = dl_manager.download_and_extract(res)
        data_dir = os.path.join(data_dir, "modelnet40_ply_hdf5_2048")
        out = []
        all_paths = tf.io.gfile.listdir(data_dir)
        json_paths = [p for p in all_paths if p.endswith(".json")]
        h5_paths = [p for p in all_paths if p.endswith(".h5")]
        for split, key in (
            (tfds.Split.TRAIN, "train"),
            (tfds.Split.TEST, "test"),
        ):
            json = [p for p in json_paths if key in p]
            h5 = [p for p in h5_paths if key in p]
            json.sort()
            h5.sort()
            paths = zip(json, h5)
            gen = tfds.core.SplitGenerator(
                name=split, gen_kwargs=dict(paths=paths, data_dir=data_dir)
            )
            out.append(gen)
        return out

    def _generate_examples(self, data_dir, paths):
        for json_path, h5_path in paths:
            json_path = os.path.join(data_dir, json_path)
            with tf.io.gfile.GFile(json_path, "rb") as fp:
                subpaths = json.load(fp)
            example_indices = [int(p.split("_")[-1][:-4]) - 1 for p in subpaths]
            h5_path = os.path.join(data_dir, h5_path)
            # with tf.io.gfile.GFile(h5_path, "r") as fp:
            # data = h5py.File(fp)
            with h5py.File(h5_path, "r") as data:
                positions = data["data"][:]
                face_indices = data["faceId"][:].astype(np.int64)
                labels = data["label"][:]
                normals = data["normal"][:]
                for i, example_index in enumerate(example_indices):
                    label = labels[i, 0]
                    yield (label, example_index), dict(
                        label=label,
                        example_index=example_index,
                        cloud=dict(
                            positions=positions[i],
                            normals=normals[i],
                            face_indices=face_indices[i],
                        ),
                    )


if __name__ == "__main__":
    config = tfds.core.download.DownloadConfig(verify_ssl=False)
    builder = Pointnet2H5()
    builder.download_and_extract(download_config=config)
