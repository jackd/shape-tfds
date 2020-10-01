"""
Original implementation:
https://github.com/charlesq34/pointnet/blob/master/provider.py
"""

import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

CITATION = """
@article{qi2016pointnet,
    title={PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation},
    author={Qi, Charles R and Su, Hao and Mo, Kaichun and Guibas, Leonidas J},
    journal={arXiv preprint arXiv:1612.00593},
    year={2016}
}
"""
URLS = ("https://github.com/charlesq34/pointnet", "https://modelnet.cs.princeton.edu/")
DL_URL = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"
NUM_POINTS = 2048
NUM_CLASSES = 40


def get_data_files(list_filename):
    with tf.io.gfile.GFile(list_filename, "r") as fp:
        lines = [line.rstrip() for line in fp]
    return lines


def load_h5(h5_filename):
    # h5py = tfds.core.lazy_imports.h5py
    import h5py

    # with tf.io.gfile.GFile(h5_filename, 'rb') as fp:
    with h5py.File(h5_filename, "r") as f:
        positions = f["data"][:]
        normals = f["normal"][:]
        face_ids = f["faceId"][:].astype(np.int64)
        labels = f["label"][:]
    labels = np.squeeze(labels, axis=1)
    return (positions, normals, face_ids, labels)


class Pointnet(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("0.0.1")

    @property
    def up_dim(self):
        return 1

    @property
    def num_classes(self):
        return NUM_CLASSES

    def _info(self):
        cloud = tfds.core.features.FeaturesDict(
            dict(
                positions=tfds.core.features.Tensor(
                    shape=(NUM_POINTS, 3), dtype=tf.float32
                ),
                normals=tfds.core.features.Tensor(
                    shape=(NUM_POINTS, 3), dtype=tf.float32
                ),
                face_ids=tfds.core.features.Tensor(shape=(NUM_POINTS,), dtype=tf.int64),
            )
        )
        return tfds.core.DatasetInfo(
            builder=self,
            description="Data used in the original pointnet paper",
            features=tfds.core.features.FeaturesDict(
                dict(
                    cloud=cloud,
                    label=tfds.core.features.ClassLabel(num_classes=NUM_CLASSES),
                )
            ),
            citation=CITATION,
            urls=URLS,
            supervised_keys=("cloud", "label"),
        )

    def _split_generators(self, dl_manager):
        base_dir = dl_manager.download_and_extract(DL_URL)
        list_path = "modelnet40_ply_hdf5_2048/{split}_files.txt"
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                num_shards=8,
                gen_kwargs=dict(
                    base_dir=base_dir, list_path=list_path.format(split="train")
                ),
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                num_shards=2,
                gen_kwargs=dict(
                    base_dir=base_dir, list_path=list_path.format(split="test")
                ),
            ),
        ]

    def _generate_examples(self, base_dir, list_path):
        data_files = get_data_files(os.path.join(base_dir, list_path))
        for i, data_file in enumerate(data_files):
            data_file = data_file[5:]  # remove leading 'data/'
            positions, normals, face_ids, labels = load_h5(
                os.path.join(base_dir, data_file)
            )
            for j, (p, n, f, l) in enumerate(zip(positions, normals, face_ids, labels)):
                cloud = dict(positions=p, normals=n, face_ids=f)
                yield (i, j), dict(cloud=cloud, label=l)
