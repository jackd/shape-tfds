from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import tensorflow_datasets as tfds
from shape_tfds.shape.modelnet.pointnet import Pointnet
import trimesh

builder = Pointnet()
# download_config = tfds.core.download.DownloadConfig(register_checksums=True)
download_config = None
builder.download_and_prepare(download_config=download_config)

for cloud, labels in builder.as_dataset(split='train', as_supervised=True):
    trimesh.PointCloud(cloud['positions'].numpy()).show()
