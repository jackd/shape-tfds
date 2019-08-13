# coding=utf-8
# Copyright 2019 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Generate shapenet-like files with random data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app

import os
import json
import random

import numpy as np
import tensorflow as tf

from tensorflow_datasets.testing import test_utils
from shape_tfds.shape.shapenet.core import load_synset_ids
from shape_tfds.shape.shapenet.part import NUM_PART_CLASSES
from shape_tfds.shape.shapenet.part import PART_SYNSET_IDS
from shape_tfds.shape.shapenet import part_test
from tensorflow_datasets.core.utils import py_utils

fake_examples_dir = os.path.join(py_utils.tfds_dir(), "testing", "test_data",
                                 "fake_examples")


def make_part_data():
    base_dir = os.path.join(
        fake_examples_dir, "shapenet_part2017",
        "shapenetcore_partanno_segmentation_benchmark_v0_normal")
    test_utils.remake_dir(base_dir)
    split_dir = os.path.join(base_dir, "train_test_split")
    tf.io.gfile.makedirs(split_dir)
    j = 0
    for split, num_examples in part_test.splits.items():
        if split == "validation":
            split = "val"
        paths = []
        synset_ids = random.sample(PART_SYNSET_IDS, num_examples)
        for synset_id in synset_ids:
            filename = 'example%d.txt' % j
            j += 1

            subdir = os.path.join(base_dir, synset_id)
            if not tf.io.gfile.isdir(subdir):
                tf.io.gfile.makedirs(subdir)
            path = os.path.join(subdir, filename)
            n_points = np.random.randint(10) + 2
            points = np.random.normal(size=n_points * 3).reshape((n_points, 3))
            normals = np.random.normal(size=n_points * 3).reshape((n_points, 3))
            normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
            point_labels = np.random.randint(NUM_PART_CLASSES, size=n_points)
            data = np.empty((n_points, 7), dtype=np.float32)
            data[:, :3] = points.astype(np.float32)
            data[:, 3:6] = normals.astype(np.float32)
            data[:, 6] = point_labels.astype(np.float32)
            with tf.io.gfile.GFile(path, "wb") as fp:
                np.savetxt(fp, data)
            paths.append(os.path.join("shape_data", synset_id, filename[:-4]))

        with tf.io.gfile.GFile(
                os.path.join(split_dir, "shuffled_%s_file_list.json" % split),
                "wb") as fp:
            json.dump(paths, fp)


def main(_):
    make_part_data()


if __name__ == "__main__":
    app.run(main)
