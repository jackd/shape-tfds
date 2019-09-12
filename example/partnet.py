from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow_datasets as tfds
from shape_tfds.shape.shapenet.partnet import Partnet

for config in Partnet.BUILDER_CONFIGS:
    Partnet(config=config).download_and_prepare()

# def vis():
#     import trimesh
#     import numpy as np
#     colors = np.array([
#         [255, 0, 0],
#         [0, 255, 0],
#         [0, 0, 255],
#         [255, 255, 0],
#         [255, 0, 255],
#         [0, 255, 255],
#     ])
#     builder = Partnet(config='chair', level=1)
#     assert (builder.num_classes[1] <= len(colors))
#     dataset = builder.as_dataset(split='train', as_supervised=True)
#     for coords, labels in tfds.as_numpy(dataset):
#         trimesh.PointCloud(coords, colors=colors[labels]).show()
#         coords[..., 1] = 0
#         trimesh.PointCloud(coords, colors=colors[labels]).show()

# vis()
